from typing import Callable, Dict, List, Union, Any, Tuple, Optional

import warnings

import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import ray
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.seed import reset_seed, log
from ray.util import PublicAPI


import torch
import pytorch_lightning as pl
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.strategies import HorovodStrategy, ParallelStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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

from ray_lightning.launchers import RayHorovodLauncher


from ray.util.check_serialize import inspect_serializability

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
        # super().__init__()
        ParallelStrategy.__init__(self, accelerator='gpu' if use_gpu else 'cpu')
        self.num_workers = num_workers
        self.cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.executor = None
        self._exit_stack = None 

        self._is_remote = False

    # def __getstate__(self):
    #     d = self.__dict__.copy()
    #     del d["executor"]
    #     return d

    # def __setstate__(self, d):
    #     d["executor"] = None
    #     self.__dict__.update(d)

    def _configure_launcher(self):
        settings = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(
            settings,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            use_gpu=self.use_gpu)

        self._launcher = RayHorovodLauncher(self, self.executor)

    def teardown(self) -> None:
        # teardown may be called before `_exit_stack` is set
        # if self._exit_stack:
        #     self._exit_stack.__exit__(None, None, None)
        #     self._exit_stack = None
        # Make sure all workers have finished training before returning to the user
        self.join()
        # self.executor.shutdown()
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
