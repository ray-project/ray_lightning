import torch

from ray.util import PublicAPI

from pytorch_lightning.strategies import HorovodStrategy, ParallelStrategy
import ray

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

from ray_lightning.launchers import RayHorovodLauncher
from ray_lightning.accelerators import \
    _GPUAccelerator  # noqa: F401


def get_executable_cls():
    # Only used for testing purposes, currently.
    # We need to override this in tests to ensure test path is set correctly.
    return None


@PublicAPI(stability="beta")
class HorovodRayStrategy(HorovodStrategy):
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
        """Initialize HorovodRayStrategy."""
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
        """Configure the Ray launcher.

        This function is overriding horovod_strategy's method.
        It is run on the driver processes.

        The horovod launcher is used to launch the Ray actors.
        """
        settings = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(
            settings,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            use_gpu=self.use_gpu)

        self._launcher = RayHorovodLauncher(self)

    @property
    def global_rank(self) -> int:
        """Return the global rank of the current process.

        This function is overriding horovod_strategy's method.
        It is run on the worker processes.
        """
        if not hvd.is_initialized():
            return 0
        return hvd.rank()

    @property
    def local_rank(self) -> int:
        """Return the local rank of the current process.

        This function is overriding horovod_strategy's method.
        It is run on the worker processes.
        """
        if not hvd.is_initialized():
            return 0
        return hvd.local_rank()

    @property
    def world_size(self) -> int:
        """Return the world size of the current process.

        This function is overriding horovod_strategy's method.
        It is run on the worker processes.
        """
        if not hvd.is_initialized():
            return self.num_workers
        return hvd.size()

    def teardown(self) -> None:
        """Teardown the strategy.

        This function is overriding horovod_strategy's method.
        It is run on the driver process.
        """
        self.join()
        self.accelerator = None
        super().teardown()

    @property
    def is_distributed(self):
        """Return whether the strategy is distributed.

        This function is a new HorovodStrategy method.
        It is run on the worker processes.
        """
        return True

    def set_remote(self, remote: bool):
        """Set the remote flag.

        This function is a new RayStrategy method.
        It is run on the worker processes.
        """
        self._is_remote = remote

    @property
    def root_device(self):
        """Return the root device.

        This function is overriding horovod_strategy's method.
        It is run on the worker processes.
        """
        if self.use_gpu and torch.cuda.is_available():
            if hvd.is_initialized():
                return torch.device("cuda", hvd.local_rank())
            else:
                return torch.device("cuda", 0)
        else:
            return torch.device("cpu")
