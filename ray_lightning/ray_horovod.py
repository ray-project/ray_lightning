import ray
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import HorovodPlugin, ParallelPlugin
from pytorch_lightning.utilities import rank_zero_only
from ray import ObjectRef

from ray_lightning import RayPlugin
from ray_lightning.session import init_session
from ray_lightning.util import process_results, Queue, Unavailable
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
        num_hosts (int): The number of nodes/machines to execute the job on.
        num_slots (int): Number of workers to be placed on each machine.
        use_gpu (bool): Whether to use GPU for allocation. For GPU to be
            used, you must also set the ``gpus`` arg in your Pytorch Lightning
            Trainer to a value > 0.

    Example:

        .. code_block:: python

            import pytorch_lightning as ptl
            from ray_lightning import HorovodRayPlugin

            ptl_model = MNISTClassifier(...)
            # 2 nodes, 4 workers per node, each using 1 CPU and 1 GPU.
            plugin = HorovodRayPlugin(num_hosts=2, num_slots=4,
                use_gpu=True)

            # If using GPUs, set the ``gpus`` arg to a value > 0.
            # The actual number of GPUs is determined by ``num_slots``.
            trainer = pl.Trainer(..., gpus=1, plugins=[plugin])
            trainer.fit(ptl_model)

    """

    def __init__(self,
                 num_hosts: int = 1,
                 num_slots: int = 1,
                 use_gpu: bool = False):
        if not HOROVOD_AVAILABLE:
            raise RuntimeError("Please intall Horovod to use this plugin.")
        if not ray.is_initialized():
            ray.init()
        super().__init__()
        self.nickname = "horovod_ray"
        self.num_hosts = num_hosts
        self.num_slots = num_slots
        self.use_gpu = use_gpu

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["executor"]
        return d

    def __setstate__(self, d):
        d["executor"] = None
        self.__dict__.update(d)

    def setup(self, model: LightningModule):
        """Creates the RayExecutor object."""
        self._model = model
        settings = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(
            settings,
            num_hosts=self.num_hosts,
            num_slots=self.num_slots,
            use_gpu=self.use_gpu)
        self.executor.start(executable_cls=get_executable_cls())

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

        results, state_dict, best_path = results[0]
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
        self.lightning_module.trainer.accelerator_connector\
            ._training_type_plugin = self
        self.lightning_module.trainer.accelerator.training_type_plugin = self

        hvd.init()
        self.global_rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
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

        return results, self.lightning_module.state_dict(), best_model_path

    def post_dispatch(self):
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
