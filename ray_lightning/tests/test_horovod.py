import pytest
import torch

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
import pytorch_lightning as pl
from ray.util.client.ray_client_helpers import ray_start_client_server

try:
    import horovod  # noqa: F401
    from horovod.common.util import nccl_built
except ImportError:
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True

import ray

from ray_lightning import HorovodRayPlugin
from ray_lightning.tests.utils import get_trainer, BoringModel, \
    train_test, load_test, predict_test, LightningMNISTClassifier


def _nccl_available():
    if not HOROVOD_AVAILABLE:
        return False
    try:
        return nccl_built()
    except AttributeError:
        return False


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    ray.shutdown()


@pytest.fixture
def ray_start_2_gpus():
    address_info = ray.init(num_cpus=2, num_gpus=2)
    yield address_info
    ray.shutdown()


@pytest.fixture
def start_ray_client_server_2_cpus():
    ray.init(num_cpus=2)
    with ray_start_client_server() as client:
        yield client


@pytest.fixture
def seed():
    pl.seed_everything(0)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_train(tmpdir, ray_start_2_cpus, seed, num_workers):
    """Tests if training modifies model weights."""
    model = BoringModel()
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=False)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    train_test(trainer, model)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_train_client(tmpdir, start_ray_client_server_2_cpus, seed,
                      num_workers):
    """Tests if training modifies model weights."""
    assert ray.util.client.ray.is_connected()
    model = BoringModel()
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=False)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    train_test(trainer, model)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_load(tmpdir, ray_start_2_cpus, seed, num_workers):
    """Tests if model checkpoint can be loaded."""
    model = BoringModel()
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=False)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    load_test(trainer, model)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_predict(tmpdir, ray_start_2_cpus, seed, num_workers):
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
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=False)
    trainer = get_trainer(
        tmpdir, limit_train_batches=20, max_epochs=1, plugins=[plugin])
    predict_test(trainer, model, dm)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_predict_client(tmpdir, start_ray_client_server_2_cpus, seed,
                        num_workers):
    assert ray.util.client.ray.is_connected()
    config = {
        "layer_1": 32,
        "layer_2": 32,
        "lr": 1e-2,
        "batch_size": 32,
    }
    model = LightningMNISTClassifier(config, tmpdir)
    dm = MNISTDataModule(
        data_dir=tmpdir, num_workers=1, batch_size=config["batch_size"])
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=False)
    trainer = get_trainer(
        tmpdir, limit_train_batches=20, max_epochs=1, plugins=[plugin])
    predict_test(trainer, model, dm)


@pytest.mark.skipif(
    not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_workers", [1, 2])
def test_train_gpu(tmpdir, ray_start_2_gpus, seed, num_workers):
    """Tests if training modifies model weights."""
    model = BoringModel()
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(tmpdir, plugins=[plugin], use_gpu=True)
    train_test(trainer, model)


@pytest.mark.skipif(
    not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_workers", [1, 2])
def test_load_gpu(tmpdir, ray_start_2_gpus, seed, num_workers):
    """Tests if model checkpoint can be loaded."""
    model = BoringModel()
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(tmpdir, plugins=[plugin], use_gpu=True)
    load_test(trainer, model)


@pytest.mark.skipif(
    not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("num_workers", [1, 2])
def test_predict_gpu(tmpdir, ray_start_2_gpus, seed, num_workers):
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
    plugin = HorovodRayPlugin(num_workers=num_workers, use_gpu=True)
    trainer = get_trainer(
        tmpdir,
        limit_train_batches=20,
        max_epochs=1,
        plugins=[plugin],
        use_gpu=True)
    predict_test(trainer, model, dm)
