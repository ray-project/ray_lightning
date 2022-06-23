import pytest
from importlib.util import find_spec
from pytorch_lightning.utilities.cli import LightningCLI
from ray_lightning import RayPlugin
from ray_lightning.tests.utils import BoringModel
from unittest import mock


@pytest.mark.skipif(
    not find_spec("jsonargparse"), reason="jsonargparse required")
def test_lightning_cli_rayplugin_instantiation():
    init_args = {
        "num_workers": 4,
        "use_gpu": False,
        "bucket_cap_mb": 50,
    }
    cli_args = ["--trainer.plugins=RayPlugin"]
    cli_args += [f"--trainer.plugins.{k}={v}" for k, v in init_args.items()]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, run=False)

    assert isinstance(cli.config_init["trainer"]["plugins"], RayPlugin)
    assert {
        k: cli.config["trainer"]["plugins"]["init_args"][k]
        for k in init_args
    } == init_args
