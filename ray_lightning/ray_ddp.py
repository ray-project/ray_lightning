from typing import Callable, Dict

import os
from collections import defaultdict

import ray
import torch
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning import _logger as log, LightningModule
from ray.util.sgd.torch.utils import setup_address

from ray_lightning.session import init_session
from ray_lightning.util import process_results, Queue
from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled


@ray.remote
class RayExecutor:
    """A class to execute any arbitrary function remotely."""

    def set_env_var(self, key: str, value: str):
        """Set an environment variable with the provided values."""
        os.environ[key] = value

    def get_node_ip(self):
        """Returns the IP address of the node that this Ray actor is on."""
        return ray.services.get_node_ip_address()

    def execute(self, fn: Callable, *args, **kwargs):
        """Execute the provided function and return the result."""
        return fn(*args, **kwargs)


class RayPlugin(DDPSpawnPlugin):
    """Pytorch Lightning plugikn for DDP training on a Ray cluster.

    This plugin is used to manage distributed training using DDP and
    Ray for process launching. Internally, the specified number of
    Ray actors are launched in the cluster and are registered as part of a
    Pytorch DDP process group. The Pytorch Lightning trainer is instantiated
    on the driver and sent to each of these training workers where training is
    executed. The distributed training protocol is handled by Pytorch DDP.

    Each training worker is configured to reserve ``num_cpus_per_worker``
    CPUS and 1 GPU if ``use_gpu`` is set to ``True``.

    If using this plugin, you should run your code like a normal Python
    script: ``python train.py``, and only on the head node if running in a
    distributed Ray cluster. There is no need to run this script on every
    single node.

    Args:
        num_workers (int): Number of training workers to use.
        num_cpus_per_worker (int): Number of CPUs per worker.
        use_gpu (bool): Whether to use GPU for allocation. For GPU to be
            used, you must also set the ``gpus`` arg in your Pytorch Lightning
            Trainer to a value > 0.
        init_hook (Callable): A function to run on each worker
            upon instantiation.

    Example:

        .. code_block:: python

            import pytorch_lightning as ptl
            from ray_lightning import RayAccelerator

            ptl_model = MNISTClassifier(...)
            plugin = RayPlugin(num_workers=4, cpus_per_worker=1,
                use_gpu=True)

            # If using GPUs, set the ``gpus`` arg to a value > 0.
            # The actual number of GPUs is determined by ``num_workers``.
            trainer = pl.Trainer(..., gpus=1, plugins=[plugin])
            trainer.fit(ptl_model)

    """

    def __init__(self,
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 init_hook: Callable = None):
        if not ray.is_initialized():
            ray.init()
        super().__init__(sync_batchnorm=None, parallel_devices=[])
        self.nickname = "ddp_ray"
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.workers = []
        self.init_hook = init_hook

    def _create_worker(self):
        """Creates Ray actor."""
        worker = RayExecutor.options(
            num_cpus=self.num_cpus_per_worker,
            num_gpus=int(self.use_gpu)).remote()
        return worker

    def setup(self, model: LightningModule):
        """Sets up PTL Trainer and creates the Ray actors."""
        # Check that trainer attribute has been set when this method is called.
        self._model = model
        self.workers = [self._create_worker() for _ in range(self.num_workers)]
        if self.init_hook:
            ray.get([w.execute.remote(self.init_hook) for w in self.workers])

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["workers"]
        return d

    def __setstate__(self, d):
        d["workers"] = []
        self.__dict__.update(d)

    def get_local_ranks(self) -> Dict[int, int]:
        """Creates a mapping of global ranks to local ranks."""
        # Get the local ranks for all the workers and store as a dict.
        # First get the IP address of each remote worker.
        node_ips = ray.get([w.get_node_ip.remote() for w in self.workers])
        rank_counter_dict = defaultdict(int)
        global_to_local = [None] * self.num_workers
        for global_rank in range(self.num_workers):
            ip = node_ips[global_rank]
            global_to_local[global_rank] = rank_counter_dict[ip]
            rank_counter_dict[ip] += 1
        return global_to_local

    def start_training(self, trainer):
        """Main training loop.

        Sets up the torch.distributed process group for each training
        worker. Then trigger remote training via ``train_remote`` on each
        worker. If using with Ray Tune, create a communication queue to
        revieve intermediate results, and process those results. Finally
        retrieve the training results from the rank 0 worker and return."""

        if "PL_GLOBAL_SEED" in os.environ:
            seed = os.environ["PL_GLOBAL_SEED"]
            ray.get([
                w.set_env_var.remote("PL_GLOBAL_SEED", seed)
                for w in self.workers
            ])

        # Get the rank 0 address for DDP connection.
        self.ddp_address = ray.get(
            self.workers[0].execute.remote(setup_address))

        self.global_to_local = self.get_local_ranks()

        model = self._model
        model_ref = ray.put(model)
        # Don't pickle the model when training remotely.
        self._model = None

        queue = None
        if TUNE_INSTALLED and is_session_enabled():
            # Create communication queue and send to all the workers.
            queue = Queue(actor_options={"num_cpus": 0})

        futures = [
            self.workers[i].execute.remote(self.train_remote, model_ref, i,
                                           queue)
            for i in range(self.num_workers)
        ]

        results = process_results(futures, queue)
        # Get the results, checkpoint path, and model weights from worker 0.
        results, best_path, state_dict = results[0]
        # Set the state for PTL using the output from remote training.
        self._results = results
        self._model = model
        self._model.load_state_dict(state_dict)
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback\
                .best_model_path = best_path

        if queue:
            # Shutdown the queue.
            queue.shutdown()

        return results

    def post_dispatch(self):
        """Shutdown the DDP process group and all the Ray actors. """

        def shutdown_remote():
            torch.distributed.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ray.get([w.execute.remote(shutdown_remote) for w in self.workers])
        for w in self.workers:
            ray.kill(w, no_restart=True)
            del w
        self.workers = []

    # All methods below are only executed in remote Ray workers.

    def train_remote(self,
                     model: LightningModule,
                     global_rank: int,
                     queue: Queue = None):
        """Training function to be executed on each remote worker."""
        assert isinstance(self, RayPlugin)
        # This method should be executed remotely in each worker.
        self._model = model
        self.lightning_module.trainer.accelerator_connector\
            ._training_type_plugin = self
        self.lightning_module.trainer.accelerator.training_type_plugin = self
        self.global_rank = global_rank

        if queue is not None:
            # Initialize session.
            init_session(rank=global_rank, queue=queue)

        # Calling new_process will call
        # transfer_distrib_spawn_state_on_fit_end.
        # We override that method and have it just set attributes.
        # Then we can just return those attributes here.
        super(RayPlugin, self).new_process(
            process_idx=global_rank,
            trainer=self.lightning_module.trainer,
            mp_queue=None)
        # Only need results from worker 0.
        if self.global_rank == 0:
            return self.results, self.best_model_path, self.model_state_dict
        else:
            return None

    def init_ddp_connection(self,
                            global_rank: int,
                            world_size: int,
                            is_slurm_managing_tasks: bool = False) -> None:
        """Process group creation to be executed on each remote worker."""
        torch_backend = "nccl" if self.use_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER:"
                     f" {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(
                backend=torch_backend,
                init_method=self.ddp_address,
                rank=global_rank,
                world_size=world_size,
            )

    def set_world_ranks(self, process_idx: int):
        """Set the appropriate rank attribues for the trainer."""
        self.local_rank = self.global_to_local[self.global_rank]
        self.global_rank = self.global_rank
        self.world_size = self.num_workers

    @property
    def root_device(self):
        # Ray already sets CUDA_VISIBLE_DEVICES for each process.
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda", 0)
        else:
            return torch.device("cpu")

    def transfer_distrib_spawn_state_on_fit_end(self, results):
        """Sets the training output as attributes so it can be retrieved."""
        if self.global_rank == 0:
            # Save training results as attributes.
            self._results = results
            self.model_state_dict = self.lightning_module.state_dict()
            best_model_path = None
            if self.lightning_module.trainer.checkpoint_callback is not None:
                best_model_path = \
                    self.lightning_module.trainer.checkpoint_callback\
                        .best_model_path
            self.best_model_path = best_model_path

    @property
    def distributed_sampler_kwargs(self):
        """Returns the args to use for torch.data.DistributedSampler."""
        distributed_sampler_kwargs = dict(
            num_replicas=self.num_workers, rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def require_distributed_sampler(self):
        """This plugin requires a distributed sampler."""
        return True

    @property
    def is_distributed(self):
        return True
