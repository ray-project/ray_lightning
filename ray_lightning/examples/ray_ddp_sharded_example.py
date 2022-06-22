import os
import tempfile
import time

import ray
import torch
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.models.vision import ImageGPT

import pytorch_lightning as pl
from pytorch_lightning import Callback

from ray_lightning import RayShardedPlugin


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = torch.tensor(
            max_memory, dtype=torch.int, device=trainer.root_gpu)
        epoch_time = torch.tensor(
            epoch_time, dtype=torch.int, device=trainer.root_gpu)

        torch.distributed.all_reduce(
            max_memory, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            epoch_time, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()

        print(
            f"Average Epoch time: {epoch_time.item() / float(world_size):.2f} "
            f"seconds")
        print(
            f"Average Peak memory {max_memory.item() / float(world_size):.2f}"
            f"MiB")


def train(data_dir, num_workers, use_gpu, batch_size, embed_dim, max_epochs,
          max_steps):
    # Make sure data is downloaded on all nodes.
    def download_data():
        from filelock import FileLock
        with FileLock(os.path.join(data_dir, ".lock")):
            MNISTDataModule(data_dir=data_dir).prepare_data()

    strategygygy = RayShardedPlugin(
        num_workers=num_workers, use_gpu=use_gpu, init_hook=download_data)

    dm = MNISTDataModule(data_dir, batch_size=batch_size)

    model = ImageGPT(
        embed_dim=embed_dim, layers=16, heads=4, vocab_size=32, num_pixels=28)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=16 if use_gpu else 32,
        callbacks=[CUDACallback()] if use_gpu else [],
        strategiesies=strategygygy,
        max_steps=max_steps)

    trainer.fit(model, dm)


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
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train for.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size to use for training.")
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=2048,
        help="Number of embedding dimensions for ImageGPT model.")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=2)
    else:
        ray.init(address=args.address)

    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")

    if args.smoke_test:
        train(
            data_dir=data_dir,
            num_workers=2,
            use_gpu=False,
            batch_size=32,
            embed_dim=16,
            max_epochs=1,
            max_steps=1)
    else:
        train(
            data_dir=data_dir,
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size,
            embed_dim=args.embed_dim,
            max_epochs=args.num_epochs,
            max_steps=None)
