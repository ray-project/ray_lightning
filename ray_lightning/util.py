import io
from typing import Callable

import torch
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning import Trainer, LightningModule

import ray


class DelayedGPUAccelerator(GPUAccelerator):
    """Same as GPUAccelerator, but doesn't do any CUDA setup.

    This allows the driver script to be launched from CPU-only machines (
    like the laptop) but have training still execute on GPU.
    """

    def setup(self, trainer: Trainer, model: LightningModule) -> None:
        return super(GPUAccelerator, self).setup(trainer, model)

    def on_train_start(self) -> None:
        if "cuda" not in str(self.root_device):
            raise RuntimeError("GPUs were requested but are not available.")
        super(DelayedGPUAccelerator, self).on_train_start()


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


def process_results(training_result_futures, queue):
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
