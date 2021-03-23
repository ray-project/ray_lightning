import os
import pytest

import ray
from ray import tune

from ray_lightning import RayPlugin, HorovodRayPlugin
from ray_lightning.tests.utils import BoringModel, get_trainer
from ray_lightning.tune import TuneReportCallback, TuneReportCheckpointCallback


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()


def train_func(dir, plugin, use_gpu=False, callbacks=None):
    def _inner_train(config):
        model = BoringModel()
        trainer = get_trainer(
            dir,
            use_gpu=use_gpu,
            callbacks=callbacks,
            plugins=[plugin],
            checkpoint_callback=False,
            **config)
        trainer.fit(model)

    return _inner_train


def tune_test(dir, plugin):
    callbacks = [TuneReportCallback(on="validation_end")]
    analysis = tune.run(
        train_func(dir, plugin, callbacks=callbacks),
        config={"max_epochs": tune.choice([1, 2, 3])},
        resources_per_trial={
            "cpu": 0,
            "extra_cpu": 2
        },
        num_samples=2)
    assert all(analysis.results_df["training_iteration"] ==
               analysis.results_df["config.max_epochs"])


def test_tune_iteration_ddp(tmpdir, ray_start_4_cpus):
    """Tests if each RayPlugin runs the correct number of iterations."""
    plugin = RayPlugin(num_workers=2, use_gpu=False)
    tune_test(tmpdir, plugin)


def test_tune_iteration_horovod(tmpdir, ray_start_4_cpus):
    """Tests if each HorovodRay trial runs the correct number of iterations."""
    plugin = HorovodRayPlugin(num_hosts=1, num_slots=2, use_gpu=False)
    tune_test(tmpdir, plugin)


def checkpoint_test(dir, plugin):
    callbacks = [TuneReportCheckpointCallback(on="validation_end")]
    analysis = tune.run(
        train_func(dir, plugin, callbacks=callbacks),
        config={"max_epochs": 2},
        resources_per_trial={
            "cpu": 0,
            "extra_cpu": 2
        },
        num_samples=1,
        local_dir=dir,
        log_to_file=True,
        metric="val_loss",
        mode="min")
    assert os.path.exists(analysis.best_checkpoint)


def test_checkpoint_ddp(tmpdir, ray_start_4_cpus):
    """Tests if Tune checkpointing works with RayAccelerator."""
    plugin = RayPlugin(num_workers=2, use_gpu=False)
    checkpoint_test(tmpdir, plugin)


def test_checkpoint_horovod(tmpdir, ray_start_4_cpus):
    """Tests if Tune checkpointing works with HorovodRayAccelerator."""
    plugin = HorovodRayPlugin(num_hosts=1, num_slots=2, use_gpu=False)
    checkpoint_test(tmpdir, plugin)
