from typing import Any, Optional, NamedTuple, Dict
from pytorch_lightning.utilities.types import _PATH
from pytorch_lightning.trainer.states import TrainerState

from contextlib import closing
import socket


def find_free_port():
    """ Find a free port on the machines. """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class _RayOutput(NamedTuple):
    """Ray output tuple with the following fields:
       - `best_model_path`: path to the best model
       - `weights_path`: path to the weights
       - `trainer_state`: trainer state
       - `trainer_results`: trainer result
       - `callback_results`: callback result
       - `logged_metrics`: logged metrics
    """
    best_model_path: Optional[_PATH]
    weights_path: Optional[_PATH]
    trainer_state: TrainerState
    trainer_results: Any
    callback_metrics: Dict[str, Any]
    logged_metrics: Dict[str, Any]
