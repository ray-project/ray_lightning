import os
import pytest
import torch

from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Callback

import ray

from ray_lightning import RayPlugin
from ray_lightning.tests.utils import get_trainer, train_test, BoringModel, \
    predict_test, LightningMNISTClassifier


@pytest.fixture
def ray_start_2_gpus():
    address_info = ray.init(num_cpus=2, num_gpus=2)
    yield address_info
    ray.shutdown()


@pytest.fixture
def seed():
    pl.seed_everything(0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_workers", [1, 2])
def test_train(tmpdir, ray_start_2_gpus, num_workers):
    """Tests if training modifies model weights."""
    model = BoringModel()
    plugin = RayPlugin(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    train_test(trainer, model)


def test_train_mixed_precision(tmpdir, ray_start_2_gpus):
    """Tests if training works with mixed precision."""
    model = BoringModel()
    plugin = RayPlugin(num_workers=2, use_gpu=True)
    trainer = get_trainer(
        tmpdir, plugins=[plugin], precision=16, amp_backend="apex")
    train_test(trainer, model)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_workers", [1, 2])
def test_predict(tmpdir, ray_start_2_gpus, seed, num_workers):
    """Tests if trained model has high accuracy on test set."""
    config = {
        "layer_1": 32,
        "layer_2": 32,
        "lr": 1e-2,
        "batch_size": 32,
    }
    model = LightningMNISTClassifier(config, tmpdir)
    dm = MNISTDataModule(
        data_dir=tmpdir, num_workers=1, batch_size=config["batch_size"])
    plugin = RayPlugin(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(
        tmpdir, limit_train_batches=20, max_epochs=1, plugins=[plugin])
    predict_test(trainer, model, dm)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_model_to_gpu(tmpdir, ray_start_2_gpus):
    """Tests if model is placed on CUDA device."""
    model = BoringModel()

    class CheckGPUCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            assert next(pl_module.parameters()).is_cuda

    plugin = RayPlugin(num_workers=2, use_gpu=True)
    trainer = get_trainer(
        tmpdir, plugins=[plugin], callbacks=[CheckGPUCallback()])
    trainer.fit(model)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_correct_devices(tmpdir, ray_start_2_gpus):
    """Tests if GPU devices are correctly set."""
    model = BoringModel()

    class CheckDevicesCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            assert trainer.root_gpu == 0
            assert int(os.environ["CUDA_VISIBLE_DEVICES"]) == \
                trainer.local_rank
            assert trainer.root_gpu == pl_module.device.index
            assert torch.cuda.current_device() == trainer.root_gpu

    plugin = RayPlugin(num_workers=2, use_gpu=True)
    trainer = get_trainer(
        tmpdir, plugins=plugin, callbacks=[CheckDevicesCallback()])
    trainer.fit(model)


@pytest.mark.skipif(
    os.environ.get("CLUSTER", "0") != "1",
    reason="Should not be run in CI. Requires multi-node Ray "
    "cluster.")
def test_multi_node(tmpdir):
    """Tests if multi-node GPU training works."""
    ray.init("auto")
    num_gpus = ray.available_resources()["GPU"]
    model = BoringModel()
    plugin = RayPlugin(num_workers=num_gpus, use_gpu=True)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    train_test(trainer, model)
