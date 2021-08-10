from pytorch_lightning.plugins import DDPSpawnShardedPlugin

from ray_lightning import RayPlugin


class RayShardedPlugin(RayPlugin, DDPSpawnShardedPlugin):
    pass
