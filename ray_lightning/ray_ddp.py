from typing import Callable, Dict, List, Union, Any, Tuple, Optional

from collections import defaultdict
from contextlib import closing
import os
import socket
import warnings

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.strategies.launchers import _SpawnLauncher
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.apply_func import move_data_to_device

import ray
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
from pytorch_lightning.utilities.seed import reset_seed, log
from ray.util import PublicAPI
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, to_state_stream, \
    load_state_stream
from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled

from pytorch_lightning.utilities.model_helpers import is_overridden

from pytorch_lightning.strategies.launchers.spawn import _FakeQueue, _SpawnOutput

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class RayLauncher(_SpawnLauncher):
    def __init__(self, strategy: "RayPlugin") -> None:
        self._strategy = strategy
        self._start_method = "ray"
        self._workers = []
        self._futures = []
        self._master_addr = None
        self._master_port = None

        self._global_to_local = None

        self.queue = None

        if not ray.is_initialized():
            ray.init()

    def is_interactive_compatible(self) -> bool:
        return True

    def launch(self,
               function: Callable,
               *args: Any,
               trainer: Optional["pl.Trainer"] = None,
               **kwargs: Any) -> Any:
        self.setup_workers()
        spawn_output = self.run_function_on_workers(
            function, *args, trainer=trainer, **kwargs)

        if trainer is None:
            return_value = spawn_output
        else:
            self._recover_results_in_main_process(spawn_output, trainer)
            return_value = spawn_output.trainer_results

        self.teardown_workers()
        return return_value

    def setup_workers(self, tune_enabled: bool = True) -> None:
        """Sets up PTL Trainer and creates the Ray actors."""
        self._workers = [
            self._create_worker() for _ in range(self._strategy.num_workers)
        ]
        if self._strategy.init_hook:
            ray.get([
                w.execute.remote(self._strategy.init_hook)
                for w in self._workers
            ])

        self._master_addr = ray.get(self._workers[0].get_node_ip.remote())
        self._master_port = str(
            ray.get(self._workers[0].execute.remote(find_free_port)))

        # Sets environment variables for all workers.
        # This will set the MASTER_ADDR and MASTER_PORT on each Ray actor.
        self._setup_env_vars()

        if self._strategy.use_gpu:
            # Set the CUDA_VISIBLE_DEVICES for all workers.
            self._share_cuda_visible_devices()

        # Get the mapping from global ranks to the respective local ranks.
        self._global_to_local = self.get_local_ranks()
        # Todo: put model into object store?

        self.tune_queue = None
        if tune_enabled and TUNE_INSTALLED and is_session_enabled():
            # Create communication queue and send to all the workers.
            self.tune_queue = Queue(actor_options={"num_cpus": 0})

    def _create_worker(self) -> ray.actor.ActorHandle:
        """Creates Ray actor."""
        worker = RayExecutor.options(
            num_cpus=self._strategy.num_cpus_per_worker,
            num_gpus=self._strategy.num_gpus_per_worker,
            resources=self._strategy.additional_resources_per_worker).remote()
        return worker

    def teardown_workers(self):
        if self.tune_queue:
            # Shutdown the queue.
            self.tune_queue.shutdown()

        for w in self._workers:
            ray.kill(w, no_restart=True)
            del w
        self._workers = []

    def get_local_ranks(self) -> List[Optional[Tuple[int, int]]]:
        """Creates a mapping of global ranks to local ranks/node ranks."""
        # Get the local ranks for all the workers and store as a list.
        # First get the IP address of each remote worker.
        node_ips = ray.get([w.get_node_ip.remote() for w in self._workers])

        node_rank_map = {}
        counter = 0
        for ip in node_ips:
            # If this is a new IP address, then increment counter.
            if ip not in node_rank_map:
                node_rank_map[ip] = counter
                counter += 1

        rank_counter_dict = defaultdict(int)
        global_to_local = [None] * self._strategy.num_workers

        for global_rank in range(self._strategy.num_workers):
            ip = node_ips[global_rank]
            global_to_local[global_rank] = (
                rank_counter_dict[ip],  # local rank
                node_rank_map[ip])  # node rank
            rank_counter_dict[ip] += 1

        return global_to_local

    def _setup_env_vars(self):
        # Get rank 0 worker address and port for DDP connection.
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = self._master_port

        # Set environment variables for remote workers.
        keys = [
            "PL_GLOBAL_SEED", "PL_TORCH_DISTRIBUTED_BACKEND", "MASTER_ADDR",
            "MASTER_PORT"
        ]
        values = [os.getenv(k) for k in keys]

        ray.get([w.set_env_vars.remote(keys, values) for w in self._workers])


    def _share_cuda_visible_devices(self):
        """Sets CUDA_VISIBLE_DEVICES on all workers.
        For each worker, CUDA_VISIBLE_DEVICES will be set to the GPU IDs
        visible to all workers on that worker's node.
        This allows GPU workers on the same node to communicate with one
        another.
        Example:
            Setup:
            - Node1:
                - Worker1: {0, 1}
                - Worker2: {2, 3}
            - Node2:
                - Worker3: {0, 1}
            CUDA_VISIBLE_DEVICES:
            - Worker1: "0,1,2,3"
            - Worker2: "0,1,2,3"
            - Worker2: "0,1"
        """
        node_ids_and_gpu_ids = ray.get(
            [w.get_node_and_gpu_ids.remote() for w in self._workers])

        node_id_to_worker_id = defaultdict(set)
        node_id_to_gpu_ids = defaultdict(set)

        for worker_id, (node_id, gpu_ids) in enumerate(node_ids_and_gpu_ids):
            node_id_to_worker_id[node_id].add(worker_id)
            node_id_to_gpu_ids[node_id].update(gpu_ids)

        futures = []
        for node_id, gpu_ids in node_id_to_gpu_ids.items():
            all_gpu_ids = ",".join([str(gpu_id) for gpu_id in gpu_ids])
            def set_gpu_ids():
                os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_ids

            for worker_id in node_id_to_worker_id[node_id]:
                futures.append(
                    self._workers[worker_id].execute.remote(set_gpu_ids))
        ray.get(futures)

    def run_function_on_workers(self,
                                function: Callable,
                                *args: Any,
                                trainer: Optional["pl.Trainer"] = None,
                                **kwargs: Any):
        self._futures = [
            w.execute.remote(self._wrapping_function, i, self._global_to_local,
                             trainer, function, args, kwargs, self.tune_queue)
            for i, w in enumerate(self._workers)
        ]

        results = process_results(self._futures, self.tune_queue)
        return results[0]

    def _wrapping_function(
            self,
            global_rank: int,
            global_to_local: List[Optional[Tuple[int, int]]],
            trainer: Optional["pl.Trainer"],
            function: Callable,
            args: Any,
            kwargs: Any,
            tune_queue: Queue,
    ) -> Any:
        self._strategy.set_remote(True)
        self._strategy.set_global_to_local(global_to_local)

        if tune_queue is not None:
            # Initialize session.
            init_session(rank=global_rank, queue=tune_queue)

        self._strategy._worker_setup(process_idx=global_rank)
        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(function.__self__, results)

        if self._strategy.local_rank == 0:
            return move_data_to_device(results, "cpu")

        return None



    def _collect_rank_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_SpawnOutput"]:
        rank_zero_debug("Finalizing the DDP spawn environment.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = trainer.lightning_module.state_dict()

        if self._strategy.global_rank != 0:
            return None


        # PyTorch Lightning saves the model weights in a temp file and
        # loads it back on the driver.
        # This won't work in a multi-node setup though, so we return the
        # model state stream directly.
        model_state_stream = to_state_stream(state_dict)
            
        # adds the `callback_metrics` to the queue
        extra = _FakeQueue()
        if is_overridden("add_to_queue", trainer.lightning_module):
            # TODO: Remove the if in v1.7
            trainer.lightning_module.add_to_queue(extra)
        self.add_to_queue(trainer, extra)

        return _SpawnOutput(best_model_path, model_state_stream, trainer.state, results, extra)



    def _recover_results_in_main_process(self, spawn_output: "_SpawnOutput", trainer: "pl.Trainer") -> None:
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback:
            trainer.checkpoint_callback.best_model_path = str(spawn_output.best_model_path)


        if spawn_output.weights_path is not None:
            state_stream = spawn_output.weights_path
            # DDPSpawnPlugin.__recover_child_process_weights begin
            # Difference here is that instead of writing the model weights to a
            # file and loading it, we use the state dict of the model directly.
            state_dict = load_state_stream(state_stream, to_gpu=self._strategy.use_gpu)
            # Set the state for PTL using the output from remote training.
            trainer.lightning_module.load_state_dict(state_dict)

        trainer.state = spawn_output.trainer_state

        # get the `callback_metrics` and set it to the trainer
        if is_overridden("get_from_queue", trainer.lightning_module):
            # only in case the user does not override it.
            # TODO: Remove the if in v1.7
            trainer.lightning_module.get_from_queue(spawn_output.extra)
        self.get_from_queue(trainer, spawn_output.extra)


@ray.remote
class RayExecutor:
    """A class to execute any arbitrary function remotely."""

    def set_env_var(self, key: str, value: str):
        """Set an environment variable with the provided values."""
        if value is not None:
            value = str(value)
            os.environ[key] = value

    def set_env_vars(self, keys: List[str], values: List[str]):
        """Sets multiple env vars with the provided values"""
        assert len(keys) == len(values)
        for key, value in zip(keys, values):
            self.set_env_var(key, value)

    def get_node_ip(self):
        """Returns the IP address of the node that this Ray actor is on."""
        return ray.util.get_node_ip_address()

    def get_node_and_gpu_ids(self):
        return ray.get_runtime_context().node_id.hex(), ray.get_gpu_ids()

    def execute(self, fn: Callable, *args, **kwargs):
        """Execute the provided function and return the result."""
        return fn(*args, **kwargs)


@PublicAPI(stability="beta")
class RayPlugin(DDPSpawnStrategy):
    """Pytorch Lightning plugin for DDP training on a Ray cluster.
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
        resources_per_worker (Optional[Dict]): If specified, the resources
            defined in this Dict will be reserved for each worker. The
            ``CPU`` and ``GPU`` keys (case-sensitive) can be defined to
            override the number of CPU/GPUs used by each worker.
        **ddp_kwargs: Additional arguments to pass into
            ``DistributedDataParallel`` initialization
    Example:
        .. code-block:: python
            import pytorch_lightning as ptl
            from ray_lightning import RayAccelerator
            ptl_model = MNISTClassifier(...)
            plugin = RayPlugin(num_workers=4, cpus_per_worker=1,
                use_gpu=True)
            # Don't set ``gpus`` in ``Trainer``.
            # The actual number of GPUs is determined by ``num_workers``.
            trainer = pl.Trainer(..., plugins=[plugin])
            trainer.fit(ptl_model)
    """

    strategy_name = "ddp_ray"

    def __init__(self,
                 num_workers: int = 1,
                 num_cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 init_hook: Optional[Callable] = None,
                 resources_per_worker: Optional[Dict] = None,
                 **ddp_kwargs: Union[Any, Dict[str, Any]]):
        resources_per_worker = resources_per_worker if resources_per_worker \
            else {}
        self.nickname = "ddp_ray"
        self.num_workers = num_workers
        self.num_cpus_per_worker = resources_per_worker.pop(
            "CPU", num_cpus_per_worker)

        if "GPU" in resources_per_worker:
            self.num_gpus_per_worker = resources_per_worker.pop("GPU")
        else:
            self.num_gpus_per_worker = int(use_gpu)

        self.use_gpu = self.num_gpus_per_worker > 0

        if self.use_gpu and self.num_gpus_per_worker < 1 and num_workers > 1:
            warnings.warn("Identified less than 1 GPU being set per worker. "
                          "If using NCCL backend (which is the default for "
                          "GPU training), GPU devices cannot be shared "
                          "across processes/workers and training is likely "
                          "to fail. It is recommended to use 1 GPU per "
                          "worker for training, or if you must use "
                          "fractional GPUs, then use the gloo backend by "
                          "setting PL_TORCH_DISTRIBUTED_BACKEND=gloo "
                          "environment variable.")

        self.additional_resources_per_worker = resources_per_worker
        self.init_hook = init_hook

        self._local_rank = 0
        self._global_rank = 0
        self._node_rank = 0

        self._is_remote = False

        super().__init__(accelerator='gpu',
            parallel_devices=[], cluster_environment=None, **ddp_kwargs)

    def _configure_launcher(self):
        self._launcher = RayLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)                

    def set_remote(self, remote: bool):
        self._is_remote = remote

    def set_global_to_local(self,
                            global_to_local: List[Optional[Tuple[int, int]]]):
        self.global_to_local = global_to_local

    def set_world_ranks(self, process_idx: int = 0):
        """Set the appropriate rank attributes for the trainer."""
        # Ranks should only be set once all the actors are created and
        # training has begun (otherwise self.global_to_local has not been
        # initialized).
        # If this method is called on the driver (i.e. self._is_remote is
        # False, then do a no-op).
        if self._is_remote:
            self._global_rank = process_idx
            self._local_rank, self._node_rank = self.global_to_local[
                self.global_rank]

    def _worker_setup(self, process_idx: int):
        reset_seed()
        self.set_world_ranks(process_idx)
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()

        # Copied from
        # pytorch_lightning.utilities.distributed.init_dist_connection
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available. "
                               "Cannot initialize distributed process group")

        if torch.distributed.is_initialized():
            log.debug(
                "torch.distributed is already initialized. Exiting early")
            return

        global_rank = self.global_rank
        world_size = self.world_size
        torch_distributed_backend = self.torch_distributed_backend

        log.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, "
                 f"MEMBER: {global_rank + 1}/{world_size}")
        torch.distributed.init_process_group(
            torch_distributed_backend, rank=global_rank, world_size=world_size, init_method='env://')

        # on rank=0 let everyone know training is starting
        rank_zero_info(f"{'-' * 100}\n"
                       f"distributed_backend={torch_distributed_backend}\n"
                       f"All distributed processes registered. "
                       f"Starting with {world_size} processes\n"
                       f"{'-' * 100}\n")

    @property
    def world_size(self) -> int:
        return self.num_workers

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @property
    def node_rank(self) -> int:
        return self._node_rank

    @property
    def root_device(self):

        if self.use_gpu and torch.cuda.is_available():
            if self._is_remote:
                # Adjust to support multiple GPUs per worker or fractional
                # GPUs per worker.
                device_id = ray.get_gpu_ids()[0]
                return torch.device("cuda", device_id)
            else:
                # If the root device is requested on the driver, just return
                # the 0th device.
                return torch.device("cuda", 0)
        else:
            return torch.device("cpu")

    @property
    def distributed_sampler_kwargs(self):
        """Returns the args to use for torch.data.DistributedSampler."""
        distributed_sampler_kwargs = dict(
            num_replicas=self.num_workers, rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self):
        return True