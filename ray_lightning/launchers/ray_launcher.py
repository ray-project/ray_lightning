from typing import Callable, List, Any, Tuple, Optional, \
    NamedTuple, Dict

from collections import defaultdict
from contextlib import closing
import os
import socket

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _Launcher
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities.apply_func import apply_to_collection,\
    move_data_to_device
from pytorch_lightning.utilities.types import _PATH
import numpy as np
import torch

import ray
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, to_state_stream, \
    load_state_stream
from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled
from pytorch_lightning.strategies import Strategy


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class RayLauncher(_Launcher):
    def __init__(self, strategy: "Strategy") -> None:
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
        ray_output = self.run_function_on_workers(
            function, *args, trainer=trainer, **kwargs)

        if trainer is None:
            return_value = ray_output
        else:
            self._recover_results_in_main_process(ray_output, trainer)
            return_value = ray_output.trainer_results

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
        function.__self__.strategy.root_device = self._strategy.root_device
        function.__self__.strategy.global_rank = self._strategy.global_rank
        # function.__self__.strategy.local_rank = self._strategy.local_rank
        self._strategy.set_cuda_device_if_used()
        
        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(function.__self__,
                                                      results)

        if self._strategy.local_rank == 0:
            return move_data_to_device(results, "cpu")

        return None

    def _collect_rank_zero_results(self, trainer: "pl.Trainer",
                                   results: Any) -> Optional["_RayOutput"]:
        rank_zero_debug("Finalizing the DDP spawn environment.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path \
            if checkpoint_callback else None

        state_dict = trainer.lightning_module.state_dict()

        if self._strategy.global_rank != 0:
            return None

        # PyTorch Lightning saves the model weights in a temp file and
        # loads it back on the driver.
        # This won't work in a multi-node setup though, so we return the
        # model state stream directly.
        model_state_stream = to_state_stream(state_dict)

        # adds the `callback_metrics`
        callback_metrics: dict = apply_to_collection(
            trainer.callback_metrics, torch.Tensor, lambda x: x.cpu().numpy(
            ))  # send as numpy to avoid issues with memory sharing

        # Same for logged_metrics
        logged_metrics: dict = apply_to_collection(
            trainer.logged_metrics, torch.Tensor, lambda x: x.cpu().numpy(
            ))  # send as numpy to avoid issues with memory sharing

        return _RayOutput(best_model_path, model_state_stream, trainer.state,
                          results, callback_metrics, logged_metrics)

    def _recover_results_in_main_process(self, ray_output: "_RayOutput",
                                         trainer: "pl.Trainer") -> None:
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback:
            trainer.checkpoint_callback.best_model_path = str(
                ray_output.best_model_path)

        if ray_output.weights_path is not None:
            state_stream = ray_output.weights_path
            # DDPSpawnPlugin.__recover_child_process_weights begin
            # Difference here is that instead of writing the model weights to a
            # file and loading it, we use the state dict of the model directly.
            state_dict = load_state_stream(
                state_stream, to_gpu=self._strategy.use_gpu)
            # Set the state for PTL using the output from remote training.
            trainer.lightning_module.load_state_dict(state_dict)

        trainer.state = ray_output.trainer_state

        trainer.callback_metrics.update(
            apply_to_collection(ray_output.callback_metrics,
                                np.ndarray, lambda x: torch.tensor(x)))
        trainer.logged_metrics.update(
            apply_to_collection(ray_output.logged_metrics,
                                np.ndarray, lambda x: torch.tensor(x)))


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


class _RayOutput(NamedTuple):
    best_model_path: Optional[_PATH]
    weights_path: Optional[_PATH]
    trainer_state: TrainerState
    trainer_results: Any
    callback_metrics: Dict[str, Any]
    logged_metrics: Dict[str, Any]
