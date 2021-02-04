import os
from collections import defaultdict

import ray
import torch
from pytorch_lightning.accelerators import DDPSpawnAccelerator
from pytorch_lightning import _logger as log
from ray.util.sgd.torch.utils import setup_address

from ray_lightning.session import init_session
from ray_lightning.util import process_results, Queue
from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled


@ray.remote
class RayExecutor:
    """A class to execute any arbitrary function remotely."""

    def set_env_var(self, key, value):
        os.environ[key] = value

    def get_node_ip(self):
        return ray.services.get_node_ip_address()

    def execute(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class RayAccelerator(DDPSpawnAccelerator):
    """Pytorch Lightning DDP Accelerator using Ray. Similar to DDP_Spawn
    accelerator except uses Ray to launch (distributed) processes instead of
    multiprocessing."""

    def __init__(self, num_workers=1, num_cpus_per_worker=1, use_gpu=False):
        super().__init__(trainer=None, nprocs=0)
        self.nickname = "ddp_ray"
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.workers = []

    def _create_worker(self):
        return RayExecutor.options(
            num_cpus=self.num_cpus_per_worker,
            num_gpus=int(self.use_gpu)).remote()

    def setup(self, model):
        # Check that trainer attribute has been set when this method is called.
        assert hasattr(self, "trainer") and self.trainer is not None
        self.trainer.use_ddp = True
        self.trainer.model = model
        self.workers = [self._create_worker() for _ in range(self.num_workers)]

    def teardown(self):
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

    def get_local_ranks(self):
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

    # All methods below are only executed in remote Ray workers.

    def train_remote(self, trainer, global_rank, queue=None):
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

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = self.global_to_local[self.global_rank]
        self.trainer.global_rank = self.global_rank
        self.trainer.world_size = self.num_workers

    def init_device(self, process_idx, is_master):
        if self.use_gpu:
            # Ray sets CUDA_VISIBLE_DEVICES already.
            gpu_idx = 0
            self.trainer.root_gpu = gpu_idx
            torch.cuda.set_device(self.trainer.root_gpu)
        else:
            pass

    def get_device_ids(self):
        if self.use_gpu:
            return super(RayAccelerator, self).get_device_ids()
        else:
            return None

    def model_to_device(self, model):
        if self.use_gpu:
            model.cuda(self.trainer.root_gpu)
        else:
            model.cpu()

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue,
                                                results):
        # Save training results as attributes.
        self.results = results
        self.model_state_dict = model.state_dict()
        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.best_model_path = best_model_path

    @property
    def distributed_sampler_kwargs(self):
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
        return True
