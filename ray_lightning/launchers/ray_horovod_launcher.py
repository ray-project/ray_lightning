from typing import Callable, List, Any, Tuple, Optional

from collections import defaultdict
from contextlib import closing
import os
import socket

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _Launcher
from pytorch_lightning.utilities.apply_func import move_data_to_device

import ray
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, Unavailable, to_state_stream, \
    load_state_stream
from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled

from pytorch_lightning.utilities.model_helpers import is_overridden

from pytorch_lightning.strategies.launchers.spawn import _FakeQueue, _SpawnOutput

try:
    import horovod.torch as hvd
    from horovod.ray import RayExecutor
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
    RayExecutor = Unavailable
    hvd = Unavailable
else:
    HOROVOD_AVAILABLE = True

from pytorch_lightning.utilities import rank_zero_only

from ray.util.check_serialize import inspect_serializability


def get_executable_cls():
    # Only used for testing purposes, currently.
    # We need to override this in tests to ensure test path is set correctly.
    return None


class RayHorovodLauncher(_Launcher):
    def __init__(self, strategy: "RayStrategy",
                 executor: "HorovodRay") -> None:
        self._strategy = strategy
        self._executor = executor

        if not ray.is_initialized():
            ray.init()
        
        self.tune_queue = None

    # def __getstate__(self):
    #     d = self.__dict__.copy()
    #     del d["executor"]
    #     return d

    # def __setstate__(self, d):
    #     d["executor"] = None
    #     self.__dict__.update(d)

    @property
    def global_rank(self) -> int:
        if not hvd.is_initialized():
            return 0
        return hvd.rank()

    @property
    def local_rank(self) -> int:
        if not hvd.is_initialized():
            return 0
        return hvd.local_rank()

    @property
    def world_size(self) -> int:
        if not hvd.is_initialized():
            return self.num_workers
        return hvd.size()

    def is_interactive_compatible(self) -> bool:
        return True

    def launch(self,
               function: Callable,
               *args: Any,
               trainer: Optional["pl.Trainer"] = None,
               **kwargs: Any) -> Any:
        spawn_output = self.run_function_on_workers(
            function, *args, trainer=trainer, **kwargs)

        if trainer is None:
            return_value = spawn_output
        else:
            self._recover_results_in_main_process(spawn_output, trainer)
            return_value = spawn_output.trainer_results

        return return_value

    def run_function_on_workers(self,
                                function: Callable,
                                *args: Any,
                                trainer: Optional["pl.Trainer"] = None,
                                **kwargs: Any):

        executor = self._executor
        self._executor = None
        self._strategy.executor = None 
        executor.start(executable_cls=get_executable_cls())

        def _func():
            return self._wrapping_function(trainer, function, args, kwargs,
                                           self.tune_queue)
        self._futures = executor.run_remote(_func)

        self._executor = executor
        self._strategy.executor = executor


        executor.shutdown()
        # results = process_results(self._futures, self.tune_queue)
        return results[0]

    def _wrapping_function(
            self,
            trainer: Optional["pl.Trainer"],
            function: Callable,
            args: Any,
            kwargs: Any,
            tune_queue: Queue,
    ) -> Any:
        self._strategy.set_remote(True)

        hvd.init()
        rank_zero_only.rank = self.global_rank

        if tune_queue is not None:
            # Initialize session.
            init_session(rank=self.global_rank, queue=tune_queue)

        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(function.__self__,
                                                      results)

        if self.local_rank == 0:
            return move_data_to_device(results, "cpu")

        return None

    def _collect_rank_zero_results(self, trainer: "pl.Trainer",
                                   results: Any) -> Optional["_SpawnOutput"]:
        rank_zero_debug("Finalizing the DDP spawn environment.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = trainer.lightning_module.state_dict()

        if self._strategy.global_rank != 0:
            return None

        # PyTorch Lightning saves the model weights in a temp file and
        # loads it back on the driver.
        # This won't work in a multi-node setup though, so we return the
        # model state stream directly.
        model_state_stream = to_state_stream(state_dict)

        # adds the `callback_metrics` to the queue
        extra = _FakeQueue()
        if is_overridden("add_to_queue", trainer.lightning_module):
            # TODO: Remove the if in v1.7
            trainer.lightning_module.add_to_queue(extra)
        self.add_to_queue(trainer, extra)

        return _SpawnOutput(best_model_path, model_state_stream, trainer.state,
                            results, extra)

    def _recover_results_in_main_process(self, spawn_output: "_SpawnOutput",
                                         trainer: "pl.Trainer") -> None:
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback:
            trainer.checkpoint_callback.best_model_path = str(
                spawn_output.best_model_path)

        if spawn_output.weights_path is not None:
            state_stream = spawn_output.weights_path
            # DDPSpawnPlugin.__recover_child_process_weights begin
            # Difference here is that instead of writing the model weights to a
            # file and loading it, we use the state dict of the model directly.
            state_dict = load_state_stream(
                state_stream, to_gpu=self._strategy.use_gpu)
            # Set the state for PTL using the output from remote training.
            trainer.lightning_module.load_state_dict(state_dict)

        trainer.state = spawn_output.trainer_state

        # get the `callback_metrics` and set it to the trainer
        if is_overridden("get_from_queue", trainer.lightning_module):
            # only in case the user does not override it.
            # TODO: Remove the if in v1.7
            trainer.lightning_module.get_from_queue(spawn_output.extra)
        self.get_from_queue(trainer, spawn_output.extra)
