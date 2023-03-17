from typing import Callable, List, Any, Tuple, Optional

from collections import defaultdict
import os

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers import _Launcher
from pytorch_lightning.utilities.apply_func import apply_to_collection,\
    move_data_to_device
import numpy as np
import torch

import ray
from ray import ObjectRef
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from ray.util.queue import Queue

from ray_lightning.session import init_session
from ray_lightning.util import process_results, to_state_stream, \
    load_state_stream, set_cuda_device_if_used
from ray_lightning.tune import TUNE_INSTALLED, is_session_enabled
from pytorch_lightning.strategies import Strategy
from ray_lightning.launchers.utils import _RayOutput, find_free_port,\
    RayExecutor


class RayLauncher(_Launcher):
    def __init__(self, strategy: "Strategy") -> None:
        """Initializes RayLauncher."""
        self._strategy = strategy
        self._start_method = "ray"
        self._workers = []
        self._futures = []
        self._master_addr = None
        self._master_port = None

        self._global_to_local = None

        self.tune_queue = None

        if not ray.is_initialized():
            ray.init()

    def is_interactive_compatible(self) -> bool:
        """Returns True if the launcher is interactive compatible."""
        return True

    def launch(self,
               function: Callable,
               *args: Any,
               trainer: Optional["pl.Trainer"] = None,
               **kwargs: Any) -> Any:
        """Launches the function on the workers from the driver node.

        This function is run on the driver process.
        """
        self.setup_workers()
        try:
            ray_output = self.run_function_on_workers(
                function, *args, trainer=trainer, **kwargs)

            if trainer is None:
                raise NotImplementedError(
                    "Ray launcher does not support trainer is None!")
            self._recover_results_in_main_process(ray_output, trainer)
            return_value = ray_output.trainer_results
        finally:
            self.teardown_workers()
            self._strategy.teardown()

        return return_value

    def setup_workers(self, tune_enabled: bool = True) -> None:
        """Creates the Ray actors and sets up PTL Trainer environment.

        This function is run on the driver process.
        """
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

        if tune_enabled and TUNE_INSTALLED and is_session_enabled():
            # Create communication queue and send to all the workers.
            self.tune_queue = Queue(actor_options={"num_cpus": 0})

    def _create_worker(self) -> ray.actor.ActorHandle:
        """Creates Ray actor workers.

        This function is run on the driver process.
        """
        worker = RayExecutor.options(
            num_cpus=self._strategy.num_cpus_per_worker,
            num_gpus=self._strategy.num_gpus_per_worker,
            resources=self._strategy.additional_resources_per_worker).remote()
        return worker

    def teardown_workers(self):
        """Tears down the Ray actors and PTL Trainer environment

        This function is run on the driver process.
        """
        if self.tune_queue:
            # Shutdown the queue.
            self.tune_queue.shutdown()

        for w in self._workers:
            ray.kill(w, no_restart=True)
            del w
        self._workers = []

    def get_local_ranks(self) -> List[Optional[Tuple[int, int]]]:
        """Creates a mapping of global ranks to local ranks/node ranks.

        This function is run on the driver process.
        """
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
        """Sets environment variables for all workers.

        This function is run on the driver process.
        """
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

        This function is run on the driver process.

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
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
        """launch a function on all workers.

        This function is run on the driver process.

        The actual training parts are run inside `_wrapping_function`
        """
        # put the model as the ray object
        # and remove the model temporarily from the args
        model = trainer.model
        model_ref = ray.put(model)
        trainer.model = None
        new_args = tuple([None] + list(args[1:]))

        # train the model and get the result to rank 0 node
        self._futures = [
            w.execute.remote(self._wrapping_function, i, self._global_to_local,
                             function, model_ref, new_args, kwargs,
                             self.tune_queue)
            for i, w in enumerate(self._workers)
        ]

        trainer.model = model

        results = process_results(self._futures, self.tune_queue)
        return results[0]

    def _wrapping_function(
            self,
            global_rank: int,
            global_to_local: List[Optional[Tuple[int, int]]],
            function: Callable,
            model_ref: ObjectRef,
            args: Any,
            kwargs: Any,
            tune_queue: Queue,
    ) -> Any:
        """Wraps the function to run on the workers.

        This function is run on the worker process.

        `results = function(*args, **kwargs)` is where the
        actual training parts are run.
        """
        self._strategy.set_remote(True)
        self._strategy.set_global_to_local(global_to_local)

        # `function` is a trainer's instance method
        # in the ray remote tasks, its bound instance `trainer`
        # will also be copied when the function is remoted.
        #
        # ALERT: passing the trainer as an argument of `_wrapping_function`
        # does not fulfill our purpose. Ray remote tasks will
        # create another copy of trainer so that
        # `function.__self__ != trainer`, in which the side effect only
        # happens to `function.__self__` when running
        # `function(*args, **kwargs)` (see SOLUTION below).
        #
        # SOLUTION: we find the trainer directly from `function`
        # by calling `function.__self__` so that we can restore
        # all the side effects happened to `function.__self__`
        trainer = function.__self__
        trainer.model = model_ref
        args = tuple([model_ref] + list(args[1:]))

        trainer._data_connector.prepare_data()
        if tune_queue is not None:
            # Initialize session.
            init_session(rank=global_rank, queue=tune_queue)

        self._strategy._worker_setup(process_idx=global_rank)
        trainer.strategy.root_device = self._strategy.root_device
        trainer.strategy.global_rank = self._strategy.global_rank
        trainer.strategy.local_rank = self._strategy.local_rank
        set_cuda_device_if_used(trainer.strategy)

        results = function(*args, **kwargs)

        if trainer is not None:
            return self._collect_rank_zero_results(trainer, results)
        else:
            return None

        trainer._teardown()
        trainer._call_teardown_hook()
        return None

    def _collect_rank_zero_results(self, trainer: "pl.Trainer",
                                   results: Any) -> Optional["_RayOutput"]:
        """Collects the results from the worker node 0.

        This function is run on the worker process.
        """
        rank_zero_debug("Finalizing the Ray launcher environment.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path \
            if checkpoint_callback else None

        state_dict = trainer.lightning_module.state_dict()

        if self._strategy.global_rank != 0:
            return None

        # Move state_dict to cpu before converting it to model state stream
        if trainer.strategy.local_rank == 0:
            state_dict = move_data_to_device(state_dict, "cpu")

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
        """Recovers the results in the main process.

        This function is run on the worker process.
        """
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
