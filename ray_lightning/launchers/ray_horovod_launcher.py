from typing import Callable, Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _Launcher
from pytorch_lightning.utilities.apply_func import apply_to_collection, \
    move_data_to_device
from torch import Tensor
import numpy as np
import torch

import ray
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from pytorch_lightning.strategies import Strategy
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, Unavailable, to_state_stream, \
    load_state_stream

from pytorch_lightning.utilities.model_helpers import is_overridden

from pytorch_lightning.strategies.launchers.spawn import _FakeQueue, \
    _SpawnOutput

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


def get_executable_cls():
    # Only used for testing purposes, currently.
    # We need to override this in tests to ensure test path is set correctly.
    return None


class RayHorovodLauncher(_Launcher):
    def __init__(self, strategy: "Strategy") -> None:
        self._strategy = strategy
        self._executor = strategy.executor

        if not ray.is_initialized():
            ray.init()

        self.tune_queue = None

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

        model = trainer.model
        model_ref = ray.put(model)
        trainer.model = None
        new_args = tuple([None] + list(args[1:]))

        executor = self._executor
        self._executor = None
        self._strategy.executor = None

        executor.start(executable_cls=get_executable_cls())

        def _func():
            return self._wrapping_function(function, model_ref, new_args,
                                           kwargs, self.tune_queue)

        self._futures = executor.run_remote(_func)

        self._executor = executor
        self._strategy.executor = executor
        trainer.model = model

        results = process_results(self._futures, self.tune_queue)
        executor.shutdown()
        return results[0]

    def _wrapping_function(
            self,
            function: Callable,
            model_ref: Any,
            args: Any,
            kwargs: Any,
            tune_queue: Queue,
    ) -> Any:
        self._strategy.set_remote(True)

        trainer = function.__self__
        model = ray.get(model_ref)
        trainer.model = model
        args = tuple([model] + list(args[1:]))

        trainer._data_connector.prepare_data()

        hvd.init()
        #         trainer.strategy.local_rank = self.local_rank
        rank_zero_only.rank = self.global_rank

        trainer.strategy.set_cuda_device_if_used()

        # Move the model to the appropriate device.
        trainer.strategy.model_to_device()

        #         trainer.strategy.setup_optimizers(trainer)
        #         trainer.strategy.setup_precision_plugin()
        #         optimizers_to_device(trainer.strategy.optimizers,\
        #            trainer.strategy.root_device)

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
        best_model_path = checkpoint_callback.best_model_path \
            if checkpoint_callback else None

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

    def add_to_queue(self, trainer: "pl.Trainer", queue: "_FakeQueue") -> None:
        """Appends the :attr:`trainer.callback_metrics` dictionary to the
        given queue. To avoid issues with memory
        sharing, we cast the data to numpy.
        Args:
            trainer: reference to the Trainer.
            queue: the instance of the queue to append the data.
        """
        callback_metrics: dict = apply_to_collection(
            trainer.callback_metrics, Tensor, lambda x: x.cpu().numpy(
            ))  # send as numpy to avoid issues with memory sharing
        queue.put(callback_metrics)

    def get_from_queue(self, trainer: "pl.Trainer",
                       queue: "_FakeQueue") -> None:
        """Retrieve the :attr:`trainer.callback_metrics` dictionary
        from the given queue. To preserve consistency,
        we cast back the data to ``torch.Tensor``.
        Args:
            trainer: reference to the Trainer.
            queue: the instance of the queue from where to get the data.
        """
        # NOTE: `add_to_queue` needs to be called before
        callback_metrics: dict = queue.get()
        trainer.callback_metrics.update(
            apply_to_collection(callback_metrics,
                                np.ndarray, lambda x: torch.tensor(x)))
