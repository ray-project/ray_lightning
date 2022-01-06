from pytorch_lightning.plugins import DDPSpawnShardedPlugin

from ray.util import PublicAPI

from ray_lightning import RayPlugin


@PublicAPI(stability="beta")
class RayShardedPlugin(RayPlugin, DDPSpawnShardedPlugin):
    pass
