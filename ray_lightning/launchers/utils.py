from typing import Any, Optional, NamedTuple, Dict, List, Callable
# from pytorch_lightning.utilities.types import _PATH
from pytorch_lightning.trainer.states import TrainerState

from contextlib import closing
import socket

import ray
import os


def find_free_port():
    """ Find a free port on the machines. """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_executable_cls():
    # Only used for testing purposes, currently.
    # Only used in `ray_horovod_launcher.py`
    # We need to override this in tests to ensure test path is set correctly.
    return None


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
        return ray.get_runtime_context().get_node_id(), ray.get_gpu_ids()

    def execute(self, fn: Callable, *args, **kwargs):
        """Execute the provided function and return the result."""
        return fn(*args, **kwargs)


class _RayOutput(NamedTuple):
    """Ray output tuple with the following fields:
       - `best_model_path`: path to the best model
       - `weights_path`: path to the weights
       - `trainer_state`: trainer state
       - `trainer_results`: trainer result
       - `callback_results`: callback result
       - `logged_metrics`: logged metrics
    """
    best_model_path: Optional[Any]
    weights_path: Optional[Any]
    trainer_state: TrainerState
    trainer_results: Any
    callback_metrics: Dict[str, Any]
    logged_metrics: Dict[str, Any]
