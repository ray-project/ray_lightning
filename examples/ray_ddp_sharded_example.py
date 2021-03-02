import pytorch_lightning as pl
from pl_bolts.models import ImageGPT
from pl_bolts.datamodules import MNISTDataModule


if __name__ == "__main__":
    #import ray
    from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin

    from ray_lightning import RayAccelerator

    #ray.init(object_store_memory=6e9)

    dm = MNISTDataModule('.')
    # 1.4B parameters.
    model = ImageGPT(embed_dim=1536, layers=48, batch_size=128, heads=8,
                     vocab_size=512)
    # trainer = pl.Trainer(accelerator=RayAccelerator(num_workers=8), plugins=[
    #     DDPShardedPlugin()])
    trainer = pl.Trainer(accelerator="ddp_cpu", num_processes=2,) #plugins=[
        #DDPShardedPlugin()])
    trainer.fit(model, dm)

