import os
import tempfile

import pytest

import ray
from ray.util.client.ray_client_helpers import ray_start_client_server


@pytest.fixture
def start_ray_client_server_2_cpus():
    ray.init(num_cpus=2)
    with ray_start_client_server() as client:
        yield client


def test_horovod_example(start_ray_client_server_2_cpus):
    assert ray.util.client.ray.is_connected()
    from ray_lightning.examples.ray_horovod_example import train_mnist
    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    config = {"layer_1": 32, "layer_2": 64, "lr": 1e-1, "batch_size": 32}
    train_mnist(
        config,
        data_dir,
        num_epochs=1,
        num_hosts=1,
        num_slots=1,
        use_gpu=False)


def test_horovod_example_tune(start_ray_client_server_2_cpus):
    assert ray.util.client.ray.is_connected()
    from ray_lightning.examples.ray_horovod_example import tune_mnist
    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    tune_mnist(
        data_dir,
        num_samples=1,
        num_epochs=1,
        num_hosts=1,
        num_slots=1,
        use_gpu=False)
