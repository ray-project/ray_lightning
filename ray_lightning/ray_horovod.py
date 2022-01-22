import torch
import pytorch_lightning as pl
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import HorovodPlugin
from pytorch_lightning.utilities import rank_zero_only

import ray
from ray import ObjectRef
from ray.util import PublicAPI
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, Unavailable, to_state_stream, \
    load_state_stream
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


def get_executable_cls():
    # Only used for testing purposes, currently.
    # We need to override this in tests to ensure test path is set correctly.
    return None


@PublicAPI(stability="beta")
class HorovodRayPlugin(HorovodPlugin):
    """Pytorch Lightning Plugin for Horovod training on a Ray cluster.

    This plugin is used to manage distributed training on a Ray cluster
    via the Horovod training framework. Internally, the specified number of
    Ray actors are launched in the cluster and are configured as part of the
    Horovod ring. The Pytorch Lightning trainer is instantiated on the
    driver and sent to each of these training workers where training is
    executed. The distributed training protocol is handled by Horovod.

    Each training worker is configured to reserve 1 CPU and if 1 GPU if
    ``use_gpu`` is set to ``True``.

    If using this plugin, you should run your code like a normal Python
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
            plugin = HorovodRayPlugin(num_workers=2, use_gpu=True)

            # Don't set ``gpus`` in ``Trainer``.
            # The actual number of GPUs is determined by ``num_workers``.
            trainer = pl.Trainer(..., plugins=[plugin])
            trainer.fit(ptl_model)

    """

    def __init__(self,
                 num_workers: int,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False):

        if not HOROVOD_AVAILABLE:
            raise RuntimeError("Please intall Horovod to use this plugin.")
        if not ray.is_initialized():
            ray.init()
        super().__init__()
        self.nickname = "horovod_ray"
        self.num_workers = num_workers
        self.cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.executor = None

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["executor"]
        return d

    def __setstate__(self, d):
        d["executor"] = None
        self.__dict__.update(d)

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

    def setup(self):
        """Creates the RayExecutor object."""
        settings = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(
            settings,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            use_gpu=self.use_gpu)
        self.executor.start(executable_cls=get_executable_cls())

    def setup_environment(self) -> None:
        # Swap out the accelerator if necessary.
        # This is needed to support CPU head with GPU workers or Ray Client.
        current_accelerator = self.lightning_module.trainer.accelerator
        if self.use_gpu and isinstance(current_accelerator, CPUAccelerator):
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

    def pre_dispatch(self):
        """All pre-dispatch logic should be done in train_remote instead."""
        pass

    def start_training(self, trainer):
        """Main training loop.

        Trigger remote training via ``train_remote`` on each
        worker. If using with Ray Tune, create a communication queue to
        retrieve intermediate results, and process those results. Finally
        retrieve the training results from the rank 0 worker and return."""
        model = self._model
        model_ref = ray.put(model)
        # Don't pickle the model when training remotely.
        self._model = None

        queue = None
        if TUNE_INSTALLED and is_session_enabled():
            # Create communication queue and send to all the workers.
            queue = Queue(actor_options={"num_cpus": 0})

        result_futures = self.executor.run_remote(
            self.train_remote, args=[model_ref, queue])

        results = process_results(result_futures, queue)

        results, state_stream, best_path = results[0]
        state_dict = load_state_stream(state_stream, to_gpu=self.use_gpu)
        self._results = results
        self._model = model
        self._model.load_state_dict(state_dict)
        self._model.trainer.accelerator.training_type_plugin = self
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback \
                .best_model_path = best_path

        if queue:
            # Shutdown the queue.
            queue.shutdown()

        return results

    def train_remote(self, model: ObjectRef, queue: Queue = None, **kwargs):
        """Training function to be executed on each remote worker."""
        self._model = ray.get(model)
        self.lightning_module.trainer._accelerator_connector\
            ._training_type_plugin = self
        self.lightning_module.trainer._accelerator_connector.accelerator\
            .training_type_plugin = self

        hvd.init()
        rank_zero_only.rank = self.global_rank

        if queue is not None:
            # Initialize session.
            init_session(rank=self.global_rank, queue=queue)

        # Move the model to the appropriate device.
        super(HorovodRayPlugin, self).model_to_device()

        # TODO: Make changes in PTL to clean this up.
        super(HorovodRayPlugin, self).pre_dispatch()
        results = super(HorovodRayPlugin,
                        self).start_training(self.lightning_module.trainer)
        if self.global_rank != 0:
            # Only want results from the first worker.
            return None

        best_model_path = None
        if self.lightning_module.trainer.checkpoint_callback is not None:
            best_model_path = \
                self.lightning_module.trainer.checkpoint_callback\
                    .best_model_path

        return results, to_state_stream(self.lightning_module.state_dict()), \
            best_model_path

    def post_dispatch(self, trainer: "pl.Trainer"):
        """Shuts down the RayExecutor."""
        self.executor.shutdown()

    @property
    def is_distributed(self):
        return True

    @property
    def root_device(self):
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda", hvd.local_rank())
        else:
            return torch.device("cpu")
