from typing import Any, Optional, NamedTuple, Dict
from pytorch_lightning.utilities.types import _PATH
from pytorch_lightning.trainer.states import TrainerState


class _RayOutput(NamedTuple):
    best_model_path: Optional[_PATH]
    weights_path: Optional[_PATH]
    trainer_state: TrainerState
    trainer_results: Any
    callback_metrics: Dict[str, Any]
    logged_metrics: Dict[str, Any]
