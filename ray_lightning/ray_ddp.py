from typing import Callable, Dict

import os
from collections import defaultdict

import ray
import torch
from pytorch_lightning.accelerators import DDPSpawnAccelerator
from pytorch_lightning import _logger as log, LightningModule, Trainer
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


class RayAccelerator(DDPSpawnAccelerator):
    """Pytorch Lightning accelerator for DDP training on a Ray cluster.

    This accelerator is used to manage distributed training using DDP and
    Ray for process launching. Internally, the specified number of
    Ray actors are launched in the cluster and are registered as part of a
    Pytorch DDP process group. The Pytorch Lightning trainer is instantiated
    on the driver and sent to each of these training workers where training is
    executed. The distributed training protocol is handled by Pytorch DDP.

    Each training worker is configured to reserve ``num_cpus_per_worker``
    CPUS and 1 GPU if ``use_gpu`` is set to ``True``.

    If using this accelerator, you should run your code like a normal Python
    script: ``python train.py``, and only on the head node if running in a
    distributed Ray cluster. There is no need to run this script on every
    single node.

    Args:
        num_workers (int): Number of training workers to use.
        num_cpus_per_worker (int): Number of CPUs per worker.
        use_gpu (bool): Whether to use GPU for allocation. For GPU to be
            used, you must also set the ``gpus`` arg in your Pytorch Lightning
            Trainer to a value > 0.

    Example:

        .. code_block:: python

            import pytorch_lightning as ptl
            from ray_lightning import RayAccelerator

            ptl_model = MNISTClassifier(...)
            accelerator = RayAccelerator(num_workers=4, cpus_per_worker=1,
                use_gpu=True)

            # If using GPUs, set the ``gpus`` arg to a value > 0.
            # The actual number of GPUs is determined by ``num_workers``.
            trainer = pl.Trainer(..., gpus=1, accelerator=accelerator)
            trainer.fit(ptl_model)

    """

    def __init__(self,
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 init_hook: Callable = None):
        super().__init__(trainer=None, nprocs=0)
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
        if self.init_hook:
            worker.execute.remote(self.init_hook)
        return worker

    def setup(self, model: LightningModule):
        """Sets up PTL Trainer and creates the Ray actors."""
        # Check that trainer attribute has been set when this method is called.
        assert hasattr(self, "trainer") and self.trainer is not None
        self.trainer.use_ddp = True
        self.trainer.model = model
        self.workers = [self._create_worker() for _ in range(self.num_workers)]

    def teardown(self):
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

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["workers"]
        return self.__dict__

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

    def train(self):
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

        trainer = self.trainer
        assert trainer is not None
        trainer_ref = ray.put(trainer)
        # Don't pickle self.trainer when training remotely.
        self.trainer = None

        queue = None
        if TUNE_INSTALLED and is_session_enabled():
            # Create communication queue and send to all the workers.
            queue = Queue(actor_options={"num_cpus": 0})

        futures = [
            self.workers[i].execute.remote(self.train_remote, trainer_ref, i,
                                           queue)
            for i in range(self.num_workers)
        ]

        results = process_results(futures, queue)
        results, best_path, state_dict = results[0]
        self.trainer = trainer
        self.trainer.model.load_state_dict(state_dict)
        if self.trainer.checkpoint_callback:
            self.trainer.checkpoint_callback.best_model_path = best_path

        if queue:
            # Shutdown the queue.
            queue.shutdown()

        return results

    # All methods below are only executed in remote Ray workers.

    def train_remote(self,
                     trainer: Trainer,
                     global_rank: int,
                     queue: Queue = None):
        """Training function to be executed on each remote worker."""
        assert isinstance(self, RayAccelerator)
        # This method should be executed remotely in each worker.
        self.trainer = trainer
        self.trainer.accelerator_backend = self
        self.global_rank = global_rank
        model = self.trainer.model

        if queue is not None:
            # Initialize session.
            init_session(rank=global_rank, queue=queue)

        # Calling ddp_train will call transfer_distrib_spawn_state_on_fit_end.
        # We override that method and have it just set attributes.
        # Then we can just return those attributes here.
        super(RayAccelerator, self).ddp_train(
            process_idx=global_rank, mp_queue=None, model=model)
        return self.results, self.best_model_path, self.model_state_dict

    def init_ddp_connection(self,
                            global_rank: int,
                            world_size: int,
                            is_slurm_managing_tasks: bool = True) -> None:
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
        self.trainer.local_rank = self.global_to_local[self.global_rank]
        self.trainer.global_rank = self.global_rank
        self.trainer.world_size = self.num_workers

    def init_device(self, process_idx: int, is_master: bool):
        """Sets the correct GPU device for the trainer and torch."""
        if self.use_gpu:
            # Ray sets CUDA_VISIBLE_DEVICES already.
            gpu_idx = 0
            self.trainer.root_gpu = gpu_idx
            torch.cuda.set_device(self.trainer.root_gpu)
        else:
            pass

    def get_device_ids(self):
        """Get the GPU device id of this worker, or None if on CPU only."""
        if self.use_gpu:
            return super(RayAccelerator, self).get_device_ids()
        else:
            return None

    def model_to_device(self, model: LightningModule):
        """Moves the model to the appropriate device."""
        if self.use_gpu:
            model.cuda(self.trainer.root_gpu)
        else:
            model.cpu()

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue,
                                                results):
        """Sets the training output as attributes so it can be retrieved."""
        # Save training results as attributes.
        self.results = results
        self.model_state_dict = model.state_dict()
        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.best_model_path = best_model_path

    @property
    def distributed_sampler_kwargs(self):
        """Returns the args to use for torch.data.DistributedSampler."""
        distributed_sampler_kwargs = dict(
            num_replicas=self.num_workers, rank=self.global_rank)
        if self.ddp_plugin is not None:
            distributed_sampler_kwargs = \
                self.ddp_plugin.distributed_sampler_kwargs(
                    distributed_sampler_kwargs
                )
        return distributed_sampler_kwargs

    @property
    def require_distributed_sampler(self):
        """This accelerator requires a distributed sampler."""
        return True
