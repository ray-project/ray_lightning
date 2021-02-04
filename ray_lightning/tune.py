from typing import Dict, List, Union

import os

from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning import Trainer, LightningModule

from ray_lightning.session import put_queue, get_actor_rank

try:
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneCallback
    from ray.tune import is_session_enabled
    TUNE_INSTALLED = True
except ImportError:
    tune = None
    TuneCallback = object

    def is_session_enabled():
        return False

    TUNE_INSTALLED = False

if TUNE_INSTALLED:

    class TuneReportCallback(TuneCallback):
        def __init__(
                self,
                metrics: Union[None, str, List[str], Dict[str, str]] = None,
                on: Union[str, List[str]] = "validation_end"):
            super(TuneReportCallback, self).__init__(on)
            if isinstance(metrics, str):
                metrics = [metrics]
            self._metrics = metrics

        def _get_report_dict(self, trainer: Trainer,
                             pl_module: LightningModule):
            # Don't report if just doing initial validation sanity checks.
            if trainer.running_sanity_check:
                return
            if not self._metrics:
                report_dict = {
                    k: v.item()
                    for k, v in trainer.callback_metrics.items()
                }
            else:
                report_dict = {}
                for key in self._metrics:
                    if isinstance(self._metrics, dict):
                        metric = self._metrics[key]
                    else:
                        metric = key
                    report_dict[key] = trainer.callback_metrics[metric].item()
            return report_dict

        def _handle(self, trainer: Trainer, pl_module: LightningModule):
            if get_actor_rank() == 0:
                report_dict = self._get_report_dict(trainer, pl_module)
                if report_dict is not None:
                    put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(TuneCallback):
        def __init__(self,
                     filename: str = "checkpoint",
                     on: Union[str, List[str]] = "validation_end"):
            super(_TuneCheckpointCallback, self).__init__(on)
            self._filename = filename

        @staticmethod
        def _create_checkpoint(checkpoint_dict: dict, global_step: int,
                               filename: str):
            with tune.checkpoint_dir(step=global_step) as checkpoint_dir:
                file_path = os.path.join(checkpoint_dir, filename)
                atomic_save(checkpoint_dict, file_path)

        def _handle(self, trainer: Trainer, pl_module: LightningModule):
            if trainer.running_sanity_check:
                return
            checkpoint_dict = trainer.checkpoint_connector.dump_checkpoint()
            global_step = trainer.global_step
            if get_actor_rank() == 0:
                put_queue(lambda: self._create_checkpoint(
                    checkpoint_dict, global_step, self._filename))

    class TuneReportCheckpointCallback(TuneCallback):
        def __init__(
                self,
                metrics: Union[None, str, List[str], Dict[str, str]] = None,
                filename: str = "checkpoint",
                on: Union[str, List[str]] = "validation_end"):
            super(TuneReportCheckpointCallback, self).__init__(on)
            self._checkpoint = _TuneCheckpointCallback(filename, on)
            self._report = TuneReportCallback(metrics, on)

        def _handle(self, trainer: Trainer, pl_module: LightningModule):
            self._checkpoint._handle(trainer, pl_module)
            self._report._handle(trainer, pl_module)
