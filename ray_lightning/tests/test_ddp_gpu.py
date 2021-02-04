import os

import pytest
import ray
import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning import Callback
from ray.tune.examples.mnist_ptl_mini import LightningMNISTClassifier

from ray_lightning import RayAccelerator
from ray_lightning.tests.utils import get_trainer, train_test, BoringModel, \
    predict_test


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
    model = BoringModel()
    accelerator = RayAccelerator(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(tmpdir, accelerator=accelerator, use_gpu=True)
    train_test(trainer, model)

@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_workers", [1, 2])
def test_predict(tmpdir, ray_start_2_gpus, seed, num_workers):
    config = {
        "layer_1": 32,
        "layer_2": 32,
        "lr": 1e-2,
        "batch_size": 32,
    }
    model = LightningMNISTClassifier(config, tmpdir)
    dm = MNISTDataModule(
        data_dir=tmpdir, num_workers=1, batch_size=config["batch_size"])
    accelerator = RayAccelerator(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(
        tmpdir, limit_train_batches=10, max_epochs=1,
        accelerator=accelerator, use_gpu=True)
    predict_test(trainer, model, dm)

@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_model_to_gpu(tmpdir, ray_start_2_gpus):
    model = BoringModel()
    class CheckGPUCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            assert next(pl_module.parameters()).is_cuda
    accelerator = RayAccelerator(num_workers=2, use_gpu=True)
    trainer = get_trainer(tmpdir, accelerator=accelerator, use_gpu=True,
                          callbacks=[CheckGPUCallback()])
    trainer.fit(model)

@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_correct_devices(tmpdir, ray_start_2_gpus):
    model = BoringModel()
    class CheckDevicesCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            assert trainer.root_gpu == 0
            assert int(os.environ["CUDA_VISIBLE_DEVICES"]) == \
                   trainer.local_rank
            assert trainer.root_gpu == pl_module.device.index
            assert torch.cuda.current_device() == trainer.root_gpu
    accelerator = RayAccelerator(num_workers=2, use_gpu=True)
    trainer = get_trainer(tmpdir, accelerator=accelerator, use_gpu=True, \
              callbacks=[CheckDevicesCallback()])
    trainer.fit(model)

@pytest.mark.skipif(os.environ.get("CLUSTER", "0") != "1",
                    reason="Should not be run in CI. Requires multi-node Ray "
                         "cluster.")
def test_multi_node(tmpdir):
    ray.init("auto")
    num_gpus = ray.available_resources()["GPU"]
    model = BoringModel()
    accelerator = RayAccelerator(num_workers=num_gpus, use_gpu=True)
    trainer = get_trainer(tmpdir, accelerator=accelerator, use_gpu=True)
    train_test(trainer, model)

