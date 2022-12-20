import os

import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer

import ray

from ray_lightning import RayShardedStrategy
from ray_lightning.tests.utils import BoringModel


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    ray.shutdown()


@pytest.fixture
def seed():
    pl.seed_everything(0)


def test_ddp_choice_sharded(tmpdir, ray_start_2_cpus, seed):
    """Tests if sharded strategy is properly recognized."""

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.strategy, RayShardedStrategy)
            raise ValueError()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        strategy=RayShardedStrategy(num_workers=2),
        callbacks=[CB()],
    )

    with pytest.raises(ValueError):
        trainer.fit(model)


def test_ddp_sharded_plugin_checkpoint(tmpdir, ray_start_2_cpus, seed):
    """Tests if checkpoint is saved correctly."""
    model = BoringModel()
    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=2),
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading.
    for ddp_param, shard_param in zip(model.parameters(),
                                      saved_model.parameters()):
        assert torch.equal(ddp_param, shard_param)


def test_ddp_sharded_plugin_finetune(tmpdir, ray_start_2_cpus, seed):
    """Tests if we can save and restart training."""
    model = BoringModel()
    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=2),
        fast_dev_run=True,
    )
    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    trainer = Trainer(fast_dev_run=True, )
    trainer.fit(saved_model)


def test_ddp_sharded_plugin_resume_from_checkpoint(tmpdir, ray_start_2_cpus,
                                                   seed):
    """Tests if resuming from checkpoint works."""
    model = BoringModel()
    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=2),
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=2),
        fast_dev_run=True,
        resume_from_checkpoint=checkpoint_path)

    trainer.fit(model)


def test_ddp_sharded_plugin_test(tmpdir, ray_start_2_cpus, seed):
    """Tests if test works without fit."""
    model = BoringModel()
    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=2),
        fast_dev_run=True,
    )

    trainer.test(model)


def test_ddp_sharded_plugin_resume_from_checkpoint_downsize(
        tmpdir, ray_start_2_cpus, seed):
    """Tests if we can save and resume training with less workers."""
    model = BoringModel()
    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=2), fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(
        strategy=RayShardedStrategy(num_workers=1),
        fast_dev_run=True,
        resume_from_checkpoint=checkpoint_path)

    trainer.fit(model)
