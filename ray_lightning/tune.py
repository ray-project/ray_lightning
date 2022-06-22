from typing import Dict, List, Union, Optional
import warnings

import fsspec
import os

from pytorch_lightning import Trainer, LightningModule
from ray.util import PublicAPI

from ray_lightning.session import put_queue, get_actor_rank
from ray_lightning.util import to_state_stream, Unavailable

try:
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneCallback
    from ray.tune import is_session_enabled
    TUNE_INSTALLED = True
except ImportError:
    tune = None
    TuneCallback = Unavailable

    def is_session_enabled():
        return False

    get_tune_resources = Unavailable

    TUNE_INSTALLED = False

if TUNE_INSTALLED:

    @PublicAPI(stability="beta")
    def get_tune_resources(
            num_workers: int = 1,
            num_cpus_per_worker: int = 1,
            use_gpu: bool = False,
            # Deprecated args.
            cpus_per_worker: Optional[int] = None,
    ) -> Dict[str, int]:
        """Returns the PlacementGroupFactory to use for Ray Tune."""
        from ray.tune import PlacementGroupFactory

        if cpus_per_worker is not None:
            # TODO(amogkam): Remove `cpus_per_worker` on next major release.
            num_cpus_per_worker = cpus_per_worker
            warnings.warn(
                "`cpus_per_worker` will be deprecated in the "
                "future. Use "
                "`num_cpus_per_worker` instead.", PendingDeprecationWarning)

        head_bundle = {"CPU": 1}
        child_bundle = {"CPU": num_cpus_per_worker, "GPU": int(use_gpu)}
        child_bundles = [child_bundle.copy() for _ in range(num_workers)]
        bundles = [head_bundle] + child_bundles
        placement_group_factory = PlacementGroupFactory(
            bundles, strategy="PACK")
        return placement_group_factory

    @PublicAPI(stability="beta")
    class TuneReportCallback(TuneCallback):
        """Distributed PyTorch Lightning to Ray Tune reporting callback

            Reports metrics to Ray Tune, specifically when training is done
            remotely with Ray via the accelerators in this library.

            Args:
                metrics (str|list|dict): Metrics to report to Tune.
                    If this is a list, each item describes the metric key
                    reported to PyTorch Lightning, and it will reported
                    under the same name to Tune. If this is a dict, each key
                    will be the name reported to Tune and the respective
                    value will be the metric key reported to PyTorch Lightning.
                on (str|list): When to trigger checkpoint creations.
                    Must be one of the PyTorch Lightning event hooks (less
                    the ``on_``), e.g. "batch_start", or "train_end".
                    Defaults to "validation_end".

            Example:

            .. code-block:: python

                import pytorch_lightning as pl
                from ray_lightning import RayStrategy
                from ray_lightning.tune import TuneReportCallback

                # Create strategygy.
                ray_plugin = RayStrategy(num_workers=4, use_gpu=True)

                # Report loss and accuracy to Tune after each validation epoch:
                trainer = pl.Trainer(strategy=[ray_plugin], callbacks=[
                    TuneReportCallback(["val_loss", "val_acc"],
                        on="validation_end")])

                # Same as above, but report as `loss` and `mean_accuracy`:
                trainer = pl.Trainer(strategy=[ray_plugin], callbacks=[
                    TuneReportCallback(
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
            if trainer.sanity_checking:
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
        """Distributed PyTorch Lightning to Ray Tune checkpoint callback

            Saves checkpoints after each validation step. To be used
            specifically with the strategies in this library.

            Checkpoint are currently not registered if no ``tune.report()``
            call is made afterwards. Consider using
            ``TuneReportCheckpointCallback`` instead.

            Args:
                filename (str): Filename of the checkpoint within the
                    checkpoint directory. Defaults to "checkpoint".
                on (str|list): When to trigger checkpoint creations.
                    Must be one of the PyTorch Lightning event hooks (less
                    the ``on_``), e.g. "batch_start", or "train_end".
                    Defaults to "validation_end".
        """

        def __init__(self,
                     filename: str = "checkpoint",
                     on: Union[str, List[str]] = "validation_end"):
            super(_TuneCheckpointCallback, self).__init__(on)
            self._filename = filename

        @staticmethod
        def _create_checkpoint(checkpoint_stream, global_step: int,
                               filename: str):
            with tune.checkpoint_dir(step=global_step) as checkpoint_dir:
                file_path = os.path.join(checkpoint_dir, filename)
                with fsspec.open(file_path, "wb") as f:
                    f.write(checkpoint_stream)

        def _handle(self, trainer: Trainer, pl_module: LightningModule):
            if trainer.sanity_checking:
                return
            checkpoint_dict = trainer.checkpoint_connector.dump_checkpoint()
            # Convert to a state stream first.
            checkpoint_stream = to_state_stream(checkpoint_dict)
            global_step = trainer.global_step
            if get_actor_rank() == 0:
                put_queue(lambda: self._create_checkpoint(
                    checkpoint_stream, global_step, self._filename))

    @PublicAPI(stability="beta")
    class TuneReportCheckpointCallback(TuneCallback):
        """PyTorch Lightning to Tune reporting and checkpointing callback.

            Saves checkpoints after each validation step. Also reports metrics
            to Tune, which is needed for checkpoint registration. To be used
            specifically with the strategies in this library.

            Args:
                metrics (str|list|dict): Metrics to report to Tune.
                    If this is a list, each item describes the metric key
                    reported to PyTorch Lightning, and it will reported
                    under the same name to Tune. If this is a dict, each key
                    will be the name reported to Tune and the respective
                    value will be the metric key reported to PyTorch Lightning.
                filename (str): Filename of the checkpoint within the
                    checkpoint directory. Defaults to "checkpoint".
                on (str|list): When to trigger checkpoint creations. Must be
                    one of the PyTorch Lightning event hooks (less the
                    ``on_``), e.g. "batch_start", or "train_end". Defaults
                    to "validation_end".


            Example:

            .. code-block:: python

                import pytorch_lightning as pl
                from ray_lightning import RayStrategy
                from ray_lightning.tune import TuneReportCheckpointCallback.

                # Create the Ray strategygy.
                ray_plugin = RayStrategy()

                # Save checkpoint after each training batch and after each
                # validation epoch.
                trainer = pl.Trainer(strategy=[ray_plugin], callbacks=[
                    TuneReportCheckpointCallback(
                        metrics={"loss": "val_loss",
                                "mean_accuracy": "val_acc"},
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

else:
    # If Tune is not installed.
    TuneReportCallback = Unavailable
    TuneReportCheckpointCallback = Unavailable
