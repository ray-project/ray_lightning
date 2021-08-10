from typing import Callable

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
