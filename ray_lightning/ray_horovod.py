import torch
from contextlib import ExitStack

from ray.util import PublicAPI

from pytorch_lightning.strategies import HorovodStrategy, ParallelStrategy
import pytorch_lightning as pl
import ray
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from ray_lightning.util import Unavailable

try:
    import horovod.torch as hvd
    from horovod.ray import RayExecutor
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
    RayExecutor = Unavailable
    hvd = Unavailable
else:
    HOROVOD_AVAILABLE = True
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from ray_lightning.launchers import RayHorovodLauncher
from ray_lightning.accelerators import \
    _GPUAccelerator  # noqa: F401

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.optimizer import optimizers_to_device


def get_executable_cls():
    # Only used for testing purposes, currently.
    # We need to override this in tests to ensure test path is set correctly.
    return None


@PublicAPI(stability="beta")
class HorovodRayStrategy(HorovodStrategy, ParallelStrategy):
    """Pytorch Lightning Strategy for Horovod training on a Ray cluster.

    This strategy is used to manage distributed training on a Ray cluster
    via the Horovod training framework. Internally, the specified number of
    Ray actors are launched in the cluster and are configured as part of the
    Horovod ring. The Pytorch Lightning trainer is instantiated on the
    driver and sent to each of these training workers where training is
    executed. The distributed training protocol is handled by Horovod.

    Each training worker is configured to reserve 1 CPU and if 1 GPU if
    ``use_gpu`` is set to ``True``.

    If using this strategy, you should run your code like a normal Python
    script: ``python train.py``, and not with ``horovodrun``.

    Args:
        num_workers (int): Number of training workers to use.
        num_cpus_per_worker (int): Number of CPUs per worker.
        use_gpu (bool): Whether to use GPU for allocation. For GPU to be
            used, you must also set the ``gpus`` arg in your Pytorch Lightning
            Trainer to a value > 0.

    Example:

        .. code-block:: python

            import pytorch_lightning as ptl
            from ray_lightning import HorovodRayPlugin

            ptl_model = MNISTClassifier(...)
            strategy = HorovodRayPlugin(num_workers=2, use_gpu=True)

            # Don't set ``gpus`` in ``Trainer``.
            # The actual number of GPUs is determined by ``num_workers``.
            trainer = pl.Trainer(..., strategy=strategy)
            trainer.fit(ptl_model)

    """
    strategy_name = "horovod_ray"

    def __init__(self,
                 num_workers: int,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False):

        if not HOROVOD_AVAILABLE:
            raise RuntimeError("Please intall Horovod to use this strategy.")
        if not ray.is_initialized():
            ray.init()
        ParallelStrategy.__init__(
            self, accelerator="_gpu" if use_gpu else "cpu")
        self.num_workers = num_workers
        self.cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.executor = None
        self._exit_stack = None
        self._local_rank = 0

        self._is_remote = False

    def _configure_launcher(self):
        settings = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(
            settings,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            use_gpu=self.use_gpu)

        self._launcher = RayHorovodLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:

        self.model_to_device()
        #         super().setup(trainer)

        self.accelerator.setup(trainer)
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        optimizers_to_device(self.optimizers, self.root_device)

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        if not trainer.training:
            # no need to setup optimizers
            return

        def _unpack_lightning_optimizer(opt):
            return opt._optimizer if isinstance(opt,
                                                LightningOptimizer) else opt

        optimizers = self.optimizers
        optimizers = [_unpack_lightning_optimizer(opt) for opt in optimizers]

        # Horovod: scale the learning rate by the number
        # of workers to account for increased total batch size
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.world_size

        # Horovod: adjust base LR used by schedulers
        # to match scaled optimizer initial LR
        lr_scheduler_configs = self.lr_scheduler_configs
        for config in lr_scheduler_configs:
            scheduler = config.scheduler
            scheduler.base_lrs = [
                lr * self.world_size for lr in scheduler.base_lrs
            ]


#         Horovod: broadcast parameters & optimizer state
#         to ensure consistent initialization
#         hvd.broadcast_parameters(self.lightning_module.state_dict(),\
#            root_rank=0)

#         for optimizer in optimizers:
#             hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        accumulation_scheduler = trainer.accumulation_scheduler
        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "Horovod currently does not support different"
                " `accumulate_grad_batches` at different epochs.")

        self.optimizers = self._wrap_optimizers(
            optimizers, trainer.accumulate_grad_batches)
        for optimizer in self.optimizers:
            # Synchronization will be performed explicitly following backward()
            self._exit_stack.enter_context(optimizer.skip_synchronize())

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

    def teardown(self) -> None:
        self.join()
        self.accelerator = None
        super().teardown()

    @property
    def is_distributed(self):
        return True

    def set_remote(self, remote: bool):
        self._is_remote = remote

    @property
    def root_device(self):
        if self.use_gpu and torch.cuda.is_available():
            if hvd.is_initialized():
                return torch.device("cuda", hvd.local_rank())
            else:
                return torch.device("cuda", 0)
        else:
            return torch.device("cpu")

    def set_cuda_device_if_used(self):
        """Set the CUDA device to use for the root node."""
        if self.use_gpu:
            # overwrite the logger
            gpu_available = True
            gpu_type = " (cuda)"
            gpu_used = True
            rank_zero_info(
                f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used} "
                "(Please ignore the previous info [GPU used: False]).")

            torch.cuda.set_device(self.root_device)
