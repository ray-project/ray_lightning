from typing import Dict, List, Optional, Union

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
    is_session_enabled = lambda: False
    TUNE_INSTALLED = False

if TUNE_INSTALLED:

    class TuneReportCallback(TuneCallback):
        """PyTorch Lightning to Ray Tune reporting callback

        Reports metrics to Ray Tune.

        Args:
            metrics (str|list|dict): Metrics to report to Tune. If this is a list,
                each item describes the metric key reported to PyTorch Lightning,
                and it will reported under the same name to Tune. If this is a
                dict, each key will be the name reported to Tune and the respective
                value will be the metric key reported to PyTorch Lightning.
            on (str|list): When to trigger checkpoint creations. Must be one of
                the PyTorch Lightning event hooks (less the ``on_``), e.g.
                "batch_start", or "train_end". Defaults to "validation_end".

        Example:

        .. code-block:: python

            import pytorch_lightning as pl
            from ray.tune.integration.pytorch_lightning import TuneReportCallback

            # Report loss and accuracy to Tune after each validation epoch:
            trainer = pl.Trainer(callbacks=[TuneReportCallback(
                    ["val_loss", "val_acc"], on="validation_end")])

            # Same as above, but report as `loss` and `mean_accuracy`:
            trainer = pl.Trainer(callbacks=[TuneReportCallback(
                    {"loss": "val_loss", "mean_accuracy": "val_acc"},
                    on="validation_end")])

        """

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
        """PyTorch Lightning checkpoint callback

        Saves checkpoints after each validation step.

        Checkpoint are currently not registered if no ``tune.report()`` call
        is made afterwards. Consider using ``TuneReportCheckpointCallback``
        instead.

        Args:
            filename (str): Filename of the checkpoint within the checkpoint
                directory. Defaults to "checkpoint".
            on (str|list): When to trigger checkpoint creations. Must be one of
                the PyTorch Lightning event hooks (less the ``on_``), e.g.
                "batch_start", or "train_end". Defaults to "validation_end".


        """

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
        """PyTorch Lightning report and checkpoint callback

        Saves checkpoints after each validation step. Also reports metrics to Tune,
        which is needed for checkpoint registration.

        Args:
            metrics (str|list|dict): Metrics to report to Tune. If this is a list,
                each item describes the metric key reported to PyTorch Lightning,
                and it will reported under the same name to Tune. If this is a
                dict, each key will be the name reported to Tune and the respective
                value will be the metric key reported to PyTorch Lightning.
            filename (str): Filename of the checkpoint within the checkpoint
                directory. Defaults to "checkpoint".
            on (str|list): When to trigger checkpoint creations. Must be one of
                the PyTorch Lightning event hooks (less the ``on_``), e.g.
                "batch_start", or "train_end". Defaults to "validation_end".


        Example:

        .. code-block:: python

            import pytorch_lightning as pl
            from ray.tune.integration.pytorch_lightning import (
                TuneReportCheckpointCallback)

            # Save checkpoint after each training batch and after each
            # validation epoch.
            trainer = pl.Trainer(callbacks=[TuneReportCheckpointCallback(
                metrics={"loss": "val_loss", "mean_accuracy": "val_acc"},
                filename="trainer.ckpt", on="validation_end")])


        """

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
