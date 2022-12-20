import pytest

import ray
import torch
from ray import tune

from ray_lightning import RayStrategy, HorovodRayStrategy
from ray_lightning.tests.utils import BoringModel, get_trainer
from ray_lightning.tune import TuneReportCallback, \
    TuneReportCheckpointCallback, get_tune_resources


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus_4_gpus():
    address_info = ray.init(num_cpus=4, num_gpus=4)
    yield address_info
    ray.shutdown()


def train_func(dir, strategy, callbacks=None):
    def _inner_train(config):
        model = BoringModel()
        trainer = get_trainer(
            dir,
            callbacks=callbacks,
            strategy=strategy,
            checkpoint_callback=False,
            **config)
        trainer.fit(model)

    return _inner_train


def tune_test(dir, strategy):
    callbacks = [TuneReportCallback(on="validation_end")]
    analysis = tune.run(
        train_func(dir, strategy, callbacks=callbacks),
        config={"max_epochs": tune.choice([1, 2, 3])},
        resources_per_trial=get_tune_resources(
            num_workers=strategy.num_workers, use_gpu=strategy.use_gpu),
        num_samples=2)
    # fix TUNE_RESULT_DELIM issue
    config_max_epochs = analysis.results_df.get("config.max_epochs", False)
    if config_max_epochs is False:
        config_max_epochs = analysis.results_df.get("config/max_epochs", False)
    assert all(analysis.results_df["training_iteration"] == config_max_epochs)


def test_tune_iteration_ddp(tmpdir, ray_start_4_cpus):
    """Tests if each RayStrategy runs the correct number of iterations."""
    strategy = RayStrategy(num_workers=2, use_gpu=False)
    tune_test(tmpdir, strategy)


def test_tune_iteration_horovod(tmpdir, ray_start_4_cpus):
    """Tests if each HorovodRay trial runs the correct number of iterations."""
    strategy = HorovodRayStrategy(num_workers=2, use_gpu=False)
    tune_test(tmpdir, strategy)


def checkpoint_test(dir, strategy):
    callbacks = [TuneReportCheckpointCallback(on="validation_end")]
    analysis = tune.run(
        train_func(dir, strategy, callbacks=callbacks),
        config={"max_epochs": 2},
        resources_per_trial=get_tune_resources(
            num_workers=strategy.num_workers, use_gpu=strategy.use_gpu),
        num_samples=1,
        local_dir=dir,
        log_to_file=True,
        metric="val_loss",
        mode="min")
    assert analysis.best_checkpoint is not None


def test_checkpoint_ddp(tmpdir, ray_start_4_cpus):
    """Tests if Tune checkpointing works with RayAccelerator."""
    strategy = RayStrategy(num_workers=2, use_gpu=False)
    checkpoint_test(tmpdir, strategy)


def test_checkpoint_horovod(tmpdir, ray_start_4_cpus):
    """Tests if Tune checkpointing works with HorovodRayAccelerator."""
    strategy = HorovodRayStrategy(num_workers=2, use_gpu=False)
    checkpoint_test(tmpdir, strategy)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="test requires multi-GPU machine")
def test_checkpoint_ddp_gpu(tmpdir, ray_start_4_cpus_4_gpus):
    """Tests if Tune checkpointing works with RayAccelerator."""
    strategy = RayStrategy(num_workers=2, use_gpu=True)
    checkpoint_test(tmpdir, strategy)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="test requires multi-GPU machine")
def test_checkpoint_horovod_gpu(tmpdir, ray_start_4_cpus_4_gpus):
    """Tests if Tune checkpointing works with HorovodRayAccelerator."""
    strategy = HorovodRayStrategy(num_workers=2, use_gpu=True)
    checkpoint_test(tmpdir, strategy)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="test requires multi-GPU machine")
def test_tune_iteration_ddp_gpu(tmpdir, ray_start_4_cpus_4_gpus):
    """Tests if each RayStrategy runs the correct number of iterations."""
    strategy = RayStrategy(num_workers=2, use_gpu=True)
    tune_test(tmpdir, strategy)
