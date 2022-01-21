import pytest
from ray.util.client.ray_client_helpers import ray_start_client_server
from torch.utils.data import DistributedSampler

from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

import ray
from ray.cluster_utils import Cluster

from ray_lightning import RayPlugin
from ray_lightning.tests.utils import get_trainer, train_test, \
    load_test, predict_test, BoringModel, LightningMNISTClassifier


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus_4_extra():
    address_info = ray.init(num_cpus=4, resources={"extra": 4})
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def start_ray_client_server_2_cpus():
    ray.init(num_cpus=2)
    with ray_start_client_server() as client:
        yield client


@pytest.fixture
def seed():
    pl.seed_everything(0)


@pytest.fixture
def ray_start_cluster_2_node_2_cpu():
    cluster = Cluster()
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    address_info = ray.init(cluster.address)
    yield address_info
    ray.shutdown()
    cluster.shutdown()


@pytest.mark.parametrize("num_workers", [1, 2])
def test_actor_creation(tmpdir, ray_start_2_cpus, num_workers):
    """Tests whether the appropriate number of training actors are created."""
    model = BoringModel()

    def check_num_actor():
        assert len(ray.state.actors()) == num_workers

    model.on_epoch_end = check_num_actor

    plugin = RayPlugin(num_workers=num_workers)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    trainer.fit(model)


def test_global_local_ranks(ray_start_4_cpus):
    """Tests local rank and node rank map is correct."""

    @ray.remote
    class Node1Actor:
        def get_node_ip(self):
            return "1"

    @ray.remote
    class Node2Actor:
        def get_node_ip(self):
            return "2"

    plugin = RayPlugin(num_workers=4, use_gpu=False)

    # 2 workers on "Node 1", 2 workers on "Node 2"
    plugin.workers = [
        Node1Actor.remote(),
        Node1Actor.remote(),
        Node2Actor.remote(),
        Node2Actor.remote()
    ]

    global_to_local = plugin.get_local_ranks()

    assert len(global_to_local) == 4
    local_ranks = {ranks[0] for ranks in global_to_local}
    node_ranks = {ranks[1] for ranks in global_to_local}

    assert local_ranks == set(range(2))
    assert node_ranks == set(range(2))

    # Make sure the rank 0 worker has local rank and node rank of 0.
    assert global_to_local[0][0] == 0
    assert global_to_local[0][1] == 0


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("extra_resource_per_worker", [1, 2])
@pytest.mark.parametrize("num_cpus_per_worker", [1, 2])
def test_actor_creation_resources(tmpdir, ray_start_4_cpus_4_extra,
                                  num_workers, extra_resource_per_worker,
                                  num_cpus_per_worker):
    """Tests if training actors are created with custom resources."""
    model = BoringModel()
    plugin = RayPlugin(
        num_workers=num_workers,
        num_cpus_per_worker=num_cpus_per_worker,
        resources_per_worker={"extra": 1})

    def check_num_actor():
        assert len(ray.state.actors()) == num_workers

    model.on_epoch_end = check_num_actor
    trainer = get_trainer(tmpdir, plugins=[plugin])
    trainer.fit(model)


def test_resource_override(ray_start_2_cpus):
    """Tests if CPU and GPU resources are overridden if manually passed in."""

    plugin = RayPlugin(num_workers=1, num_cpus_per_worker=2, use_gpu=True)
    assert plugin.num_cpus_per_worker == 2
    assert plugin.use_gpu

    plugin = RayPlugin(
        num_workers=1,
        num_cpus_per_worker=2,
        use_gpu=True,
        resources_per_worker={"CPU": 3})
    assert plugin.num_cpus_per_worker == 3
    assert plugin.use_gpu

    plugin = RayPlugin(
        num_workers=1,
        num_cpus_per_worker=2,
        use_gpu=True,
        resources_per_worker={"GPU": 0})
    assert plugin.num_cpus_per_worker == 2
    assert not plugin.use_gpu

    plugin = RayPlugin(
        num_workers=1,
        num_cpus_per_worker=2,
        use_gpu=False,
        resources_per_worker={"GPU": 1})
    assert plugin.num_cpus_per_worker == 2
    assert plugin.use_gpu

    plugin = RayPlugin(
        num_workers=1,
        num_cpus_per_worker=2,
        use_gpu=False,
        resources_per_worker={"GPU": 2})
    assert plugin.num_cpus_per_worker == 2
    assert plugin.num_gpus_per_worker == 2
    assert plugin.use_gpu


def test_distributed_sampler(tmpdir, ray_start_2_cpus):
    """Tests if distributed sampler is properly set."""
    model = BoringModel()
    train_dataloader = model.train_dataloader()
    initial_sampler = train_dataloader.sampler
    assert not isinstance(initial_sampler, DistributedSampler)

    class DistributedSamplerCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            train_sampler = trainer.train_dataloader.sampler
            assert isinstance(train_sampler, DistributedSampler)
            assert train_sampler.shuffle
            assert train_sampler.num_replicas == 2
            assert train_sampler.rank == trainer.global_rank

        def on_validation_start(self, trainer, pl_module):
            train_sampler = trainer.val_dataloaders[0].sampler
            assert isinstance(train_sampler, DistributedSampler)
            assert not train_sampler.shuffle
            assert train_sampler.num_replicas == 2
            assert train_sampler.rank == trainer.global_rank

        def on_test_start(self, trainer, pl_module):
            train_sampler = trainer.test_dataloaders[0].sampler
            assert isinstance(train_sampler, DistributedSampler)
            assert not train_sampler.shuffle
            assert train_sampler.num_replicas == 2
            assert train_sampler.rank == trainer.global_rank

    plugin = RayPlugin(num_workers=2)
    trainer = get_trainer(
        tmpdir, plugins=[plugin], callbacks=[DistributedSamplerCallback()])
    trainer.fit(model)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_train(tmpdir, ray_start_2_cpus, num_workers):
    """Tests if training modifies model weights."""
    model = BoringModel()
    plugin = RayPlugin(num_workers=num_workers)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    train_test(trainer, model)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_train_client(tmpdir, start_ray_client_server_2_cpus, num_workers):
    assert ray.util.client.ray.is_connected()
    model = BoringModel()
    plugin = RayPlugin(num_workers=num_workers)
    trainer = get_trainer(tmpdir, plugins=[plugin])
    train_test(trainer, model)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_load(tmpdir, ray_start_2_cpus, num_workers):
    """Tests if model checkpoint can be loaded."""
    model = BoringModel()
    plugin = RayPlugin(num_workers=num_workers, use_gpu=False)
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
    plugin = RayPlugin(num_workers=num_workers, use_gpu=False)
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
    plugin = RayPlugin(num_workers=num_workers, use_gpu=False)
    trainer = get_trainer(
        tmpdir, limit_train_batches=20, max_epochs=1, plugins=[plugin])
    predict_test(trainer, model, dm)


def test_early_stop(tmpdir, ray_start_2_cpus):
    """Tests if early stopping callback works correctly."""
    model = BoringModel()
    plugin = RayPlugin(num_workers=1, use_gpu=False)
    patience = 2
    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, verbose=True)
    trainer = get_trainer(
        tmpdir,
        max_epochs=500,
        plugins=[plugin],
        callbacks=[early_stop],
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        progress_bar_refresh_rate=1)
    trainer.fit(model)
    trained_model = BoringModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)
    assert trained_model.val_epoch == patience + 1, trained_model.val_epoch


def test_unused_parameters(tmpdir, ray_start_2_cpus):
    """Tests if find_unused_parameters is properly passed to model."""
    model = BoringModel()
    plugin = RayPlugin(
        num_workers=2, use_gpu=False, find_unused_parameters=False)

    class UnusedParameterCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.model.find_unused_parameters is False

    trainer = get_trainer(
        tmpdir, plugins=[plugin], callbacks=[UnusedParameterCallback()])
    trainer.fit(model)
