from typing import Callable, Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _Launcher
from lightning_utilities.core.apply_func import apply_to_collection
import numpy as np
import torch

import ray
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from pytorch_lightning.strategies import Strategy
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, Unavailable, to_state_stream, \
    load_state_stream, set_cuda_device_if_used

from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled

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
from ray_lightning.accelerators import \
    _GPUAccelerator  # noqa: F401
from ray_lightning.launchers.utils import _RayOutput, get_executable_cls


class RayHorovodLauncher(_Launcher):
    def __init__(self, strategy: "Strategy") -> None:
        """Initialize the Ray horovod launcher."""
        self._strategy = strategy
        self._executor = strategy.executor

        if not ray.is_initialized():
            ray.init()

        self.tune_queue = None

    @property
    def global_rank(self) -> int:
        """Return the global rank of the current process.

        This function is run on the worker process.
        """
        if not hvd.is_initialized():
            return 0
        return hvd.rank()

    @property
    def local_rank(self) -> int:
        """Return the local rank of the current process.

        This function is run on the worker process.
        """
        if not hvd.is_initialized():
            return 0
        return hvd.local_rank()

    @property
    def world_size(self) -> int:
        """Return the world size of the current process.

        This function is run on the worker process.
        """
        if not hvd.is_initialized():
            return self.num_workers
        return hvd.size()

    def is_interactive_compatible(self) -> bool:
        """Return whether the launcher is interactive compatible."""
        return True

    def launch(self,
               function: Callable,
               *args: Any,
               trainer: Optional["pl.Trainer"] = None,
               **kwargs: Any) -> Any:
        """Launch the function on the workers and collect the results.

        This function is run on the driver process.
        """
        ray_output = self.run_function_on_workers(
            function, *args, trainer=trainer, **kwargs)

        if trainer is None:
            raise NotImplementedError(
                "Ray launcher does not support trainer is None! "
                "Did you override the `trainer` variable? "
                "If not, please help file an issue on Github.")
        self._recover_results_in_main_process(ray_output, trainer)
        return_value = ray_output.trainer_results

        return return_value

    def run_function_on_workers(self,
                                function: Callable,
                                *args: Any,
                                trainer: Optional["pl.Trainer"] = None,
                                **kwargs: Any):
        """Run the function on the workers and collect the results.

        This function is run on the driver process.

        `executor.run_remote` is used to launch multiple ray remote tasks
        to distributed training the model using the horovod backend.
        """

        # put the model as the ray object
        # this reduce the memory comsumption
        # and remove the model temporarily from the args
        model = trainer.model
        model_ref = ray.put(model)
        trainer.model = None
        # the model always be at the 0th position in the args
        new_args = tuple([None] + list(args[1:]))

        # remove the executor temporarily from the args
        # in order to avoid the ray.get() call in the function
        # because executor is not pickleable
        executor = self._executor
        self._executor = None
        self._strategy.executor = None

        executor.start(executable_cls=get_executable_cls())

        if TUNE_INSTALLED and is_session_enabled():
            # Create communication queue and send to all the workers.
            self.tune_queue = Queue(actor_options={"num_cpus": 0})

        self._futures = executor.run_remote(lambda: self._wrapping_function(
            function, model_ref, new_args, kwargs, self.tune_queue))

        # put back the executor and model
        self._executor = executor
        self._strategy.executor = executor
        trainer.model = model

        results = process_results(self._futures, self.tune_queue)
        executor.shutdown()
        self._strategy.teardown()

        return results[0]

    def _wrapping_function(
            self,
            function: Callable,
            model_ref: Any,
            args: Any,
            kwargs: Any,
            tune_queue: Queue,
    ) -> Any:
        """Wrapping function to run the function on the workers.

        This function is run on the worker process.

        `_wrapping_function` is run on each remote worker.
        `function(*args, **kwargs)` is where the actual training happens.
        """

        self._strategy.set_remote(True)

        # `function` is a trainer's instance method
        # in the ray remote tasks, its bound instance `trainer`
        # will also be copied when the function is remoted.
        #
        # ALERT: passing the trainer as an argument of `_wrapping_function`
        # does not fulfill our purpose. Ray remote tasks will
        # create another copy of trainer so that
        # `function.__self__ != trainer`, in which the side effect only
        # happens to `function.__self__` when running
        # `function(*args, **kwargs)` (see SOLUTION below).
        #
        # SOLUTION: we find the trainer directly from `function`
        # by calling `function.__self__` so that we can restore
        # all the side effects happened to `function.__self__`
        trainer = function.__self__
        model = ray.get(model_ref)
        trainer.model = model
        args = tuple([model] + list(args[1:]))

        trainer._data_connector.prepare_data()

        hvd.init()
        rank_zero_only.rank = self.global_rank
        set_cuda_device_if_used(trainer.strategy)

        # Move the model to the appropriate device.
        trainer.strategy.model_to_device()

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
                                   results: Any) -> Optional["_RayOutput"]:
        """Collect the results from the rank zero process.

        This function is run on the worker process.
        """
        rank_zero_debug("Finalizing the ray horovod launcher environment.")
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

        # adds the `callback_metrics`
        callback_metrics: dict = apply_to_collection(
            trainer.callback_metrics, torch.Tensor, lambda x: x.cpu().numpy(
            ))  # send as numpy to avoid issues with memory sharing

        # Same for logged_metrics
        logged_metrics: dict = apply_to_collection(
            trainer.logged_metrics, torch.Tensor, lambda x: x.cpu().numpy(
            ))  # send as numpy to avoid issues with memory sharing

        return _RayOutput(best_model_path, model_state_stream, trainer.state,
                          results, callback_metrics, logged_metrics)

    def _recover_results_in_main_process(self, ray_output: "_RayOutput",
                                         trainer: "pl.Trainer") -> None:
        """Recover the results in the main process.

        This function is run on the worker process.
        """
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback:
            trainer.checkpoint_callback.best_model_path = str(
                ray_output.best_model_path)

        if ray_output.weights_path is not None:
            state_stream = ray_output.weights_path
            # DDPSpawnPlugin.__recover_child_process_weights begin
            # Difference here is that instead of writing the model weights to a
            # file and loading it, we use the state dict of the model directly.
            state_dict = load_state_stream(
                state_stream, to_gpu=self._strategy.use_gpu)
            # Set the state for PTL using the output from remote training.
            trainer.lightning_module.load_state_dict(state_dict)

        trainer.state = ray_output.trainer_state

        trainer.callback_metrics.update(
            apply_to_collection(ray_output.callback_metrics,
                                np.ndarray, lambda x: torch.tensor(x)))
        trainer.logged_metrics.update(
            apply_to_collection(ray_output.logged_metrics,
                                np.ndarray, lambda x: torch.tensor(x)))
