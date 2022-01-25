import abc
import io
from typing import Callable

import torch
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning import Trainer

import ray


class DelayedGPUAccelerator(GPUAccelerator):
    """Same as GPUAccelerator, but doesn't do any CUDA setup.

    This allows the driver script to be launched from CPU-only machines (
    like the laptop) but have training still execute on GPU.
    """

    def setup_environment(self) -> None:
        # Don't do any CUDA setup.
        # Directly call the setup_environment method of the superclass of
        # GPUAccelerator.
        super(GPUAccelerator, self).setup_environment()

    def setup(
            self,
            trainer: Trainer,
    ) -> None:
        # Don't do any CUDA setup.
        # Directly call the setup_environment method of the superclass of
        # GPUAccelerator.
        return super(GPUAccelerator, self).setup(trainer)

    def on_train_start(self) -> None:
        if "cuda" not in str(self.root_device):
            raise RuntimeError("GPUs were requested but are not available.")
        torch.cuda.set_device(self.root_device)
        super(DelayedGPUAccelerator, self).on_train_start()

class DelayedGPUAcceleratorMixin(abc.ABC):
    def setup_environment(self) -> None:
        # Swap out the accelerator if necessary.
        # This is needed to support CPU head with GPU workers or Ray Client.
        current_accelerator = self.lightning_module.trainer.accelerator

        if self.use_gpu:
            from weakref import proxy
            from ray_lightning.util import DelayedGPUAccelerator
            precision_plugin = current_accelerator.precision_plugin
            new_accelerator = DelayedGPUAccelerator(
                precision_plugin=precision_plugin, training_type_plugin=self)
            self.lightning_module.trainer._accelerator_connector \
                ._training_type_plugin = \
                proxy(new_accelerator.training_type_plugin)
            self.lightning_module.trainer._accelerator_connector \
                ._precision_plugin = proxy(new_accelerator.precision_plugin)
            self.lightning_module.trainer._accelerator_connector.accelerator \
                = new_accelerator



class Unavailable:
    """No object should be instance of this class"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("This class should never be instantiated.")


def _handle_queue(queue):
    """Process results from the queue."""
    while not queue.empty():
        (actor_rank, item) = queue.get()
        if isinstance(item, Callable):
            item()


def process_results(training_result_futures, queue=None):
    """Process results from the queue, and return results from the futures."""
    not_ready = training_result_futures
    while not_ready:
        if queue:
            _handle_queue(queue)
        ready, not_ready = ray.wait(not_ready, timeout=0)
        ray.get(ready)
    ray.get(ready)

    if queue:
        # Process any remaining items in queue.
        _handle_queue(queue)
    return ray.get(training_result_futures)


def to_state_stream(model_state_dict):
    """Converts the given state dict to a stream of bytes."""
    _buffer = io.BytesIO()
    torch.save(model_state_dict, _buffer)
    return _buffer.getvalue()


def load_state_stream(state_stream, to_gpu):
    """Converts the state stream to a state dict on the appropriate device.

    Converts to GPU if ``to_gpu`` is True and CUDA is available.

    """
    _buffer = io.BytesIO(state_stream)
    to_gpu = to_gpu and torch.cuda.is_available()
    state_dict = torch.load(
        _buffer,
        map_location=("cpu"
                      if not to_gpu else lambda storage, loc: storage.cuda()))
    return state_dict
