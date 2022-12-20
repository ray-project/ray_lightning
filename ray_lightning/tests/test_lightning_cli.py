import pytest
from importlib.util import find_spec
from pytorch_lightning.utilities.cli import LightningCLI
from ray_lightning import RayStrategy
from ray_lightning.tests.utils import BoringModel
from unittest import mock


@pytest.mark.skipif(
    not find_spec("jsonargparse"), reason="jsonargparse required")
def test_lightning_cli_raystrategy_instantiation():
    init_args = {
        "num_workers": 4,  # Resolve from RayStrategy.__init__
        "use_gpu": False,  # Resolve from RayStrategy.__init__
        "bucket_cap_mb": 50,  # Resolve from DistributedDataParallel.__init__
    }
    cli_args = ["--trainer.strategy=RayStrategy"]
    cli_args += [f"--trainer.strategy.{k}={v}" for k, v in init_args.items()]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, run=False)

    assert isinstance(cli.config_init["trainer"]["strategy"], RayStrategy)
    assert {
        k: cli.config["trainer"]["strategy"]["init_args"][k]
        for k in init_args
    } == init_args
