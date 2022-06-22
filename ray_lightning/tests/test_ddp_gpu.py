import os
import pytest
import torch

from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Callback

import ray

from ray_lightning import RayStrategy
from ray_lightning.tests.utils import get_trainer, train_test, BoringModel, \
    predict_test, LightningMNISTClassifier


@pytest.fixture
def ray_start_2_gpus():
    address_info = ray.init(num_cpus=2, num_gpus=2)
    yield address_info
    ray.shutdown()


@pytest.fixture
def ray_start_4_gpus():
    address_info = ray.init(num_cpus=4, num_gpus=4)
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
    strategygy = RayStrategy(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(tmpdir, strategy=strategygy)
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
    strategygy = RayStrategy(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(
        tmpdir, limit_train_batches=20, max_epochs=1, strategy=[strategygy])
    predict_test(trainer, model, dm)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_model_to_gpu(tmpdir, ray_start_2_gpus):
    """Tests if model is placed on CUDA device."""
    model = BoringModel()

    class CheckGPUCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            assert next(pl_module.parameters()).is_cuda

    strategygy = RayStrategy(num_workers=2, use_gpu=True)
    trainer = get_trainer(
        tmpdir, strategy=[strategygy], callbacks=[CheckGPUCallback()])
    trainer.fit(model)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_gpus_per_worker", [0.4, 0.5, 1, 2])
def test_correct_devices(tmpdir, ray_start_4_gpus, num_gpus_per_worker,
                         monkeypatch):
    """Tests if GPU devices are correctly set."""
    model = BoringModel()

    if num_gpus_per_worker < 1:
        monkeypatch.setenv("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")

    def get_gpu_placement(current_worker_index, num_gpus_per_worker):
        """Simulates GPU resource bin packing."""
        next_gpu_index = 0
        starting_resource_count = num_gpus_per_worker
        for _ in range(current_worker_index + 1):
            current_gpu_index = next_gpu_index
            next_resources = starting_resource_count + \
                num_gpus_per_worker - 0.0001
            # If the next worker cannot fit on the current GPU, then we move
            # onto the next GPU.
            if int(next_resources) != current_gpu_index:
                increment = max(1, int(num_gpus_per_worker))
                next_gpu_index = current_gpu_index + increment

        return current_gpu_index

    class CheckDevicesCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            assert trainer.root_gpu == get_gpu_placement(
                trainer.local_rank, num_gpus_per_worker)
            assert trainer.root_gpu == pl_module.device.index
            assert torch.cuda.current_device() == trainer.root_gpu

    strategygy = RayStrategy(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={"GPU": num_gpus_per_worker})
    trainer = get_trainer(
        tmpdir, strategy=[strategygy], callbacks=[CheckDevicesCallback()])
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
    strategygy = RayStrategy(num_workers=num_gpus, use_gpu=True)
    trainer = get_trainer(tmpdir, strategy=[strategygy])
    train_test(trainer, model)
