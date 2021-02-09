"""Example using Pytorch Lightning with Pytorch DDP on Ray Accelerator."""
import os
import tempfile

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

import pytorch_lightning as pl
import ray
from ray import tune
from ray.tune.examples.mnist_ptl_mini import LightningMNISTClassifier
from ray_lightning.tune import TuneReportCallback
from ray_lightning import RayAccelerator


def download_data():
    from filelock import FileLock
    with FileLock("./data"):
        dm = MNISTDataModule(data_dir=data_dir, num_workers=1, batch_size=1)
        dm.prepare_data()
    return


def train_mnist(config,
                data_dir=None,
                num_epochs=10,
                num_workers=1,
                use_gpu=False,
                callbacks=None):
    model = LightningMNISTClassifier(config, data_dir)

    callbacks = callbacks or []

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=int(use_gpu),
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        accelerator=RayAccelerator(
            num_workers=num_workers, use_gpu=use_gpu, init_hook=download_data))
    dm = MNISTDataModule(
        data_dir=data_dir, num_workers=1, batch_size=config["batch_size"])
    trainer.fit(model, dm)


def tune_mnist(data_dir,
               num_samples=10,
               num_epochs=10,
               num_workers=1,
               use_gpu=False):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # Add Tune callback.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=callbacks)
    analysis = tune.run(
        trainable,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        resources_per_trial={
            "cpu": 1,
            "gpu": int(use_gpu),
            "extra_cpu": num_workers,
            "extra_gpu": num_workers * int(use_gpu)
        },
        name="tune_mnist")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of training workers to use.",
        default=1)
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for training.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to tune.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train for.")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    args, _ = parser.parse_known_args()

    num_epochs = 1 if args.smoke_test else args.num_epochs
    num_workers = 1 if args.smoke_test else args.num_workers
    use_gpu = False if args.smoke_test else args.use_gpu
    num_samples = 1 if args.smoke_test else args.num_samples

    if args.smoke_test:
        ray.init(num_cpus=2)
    else:
        ray.init(address=args.address)

    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    tune_mnist(data_dir, num_samples, num_epochs, num_workers, use_gpu)
