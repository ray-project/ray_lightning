from pytorch_lightning.strategies import DDPSpawnShardedStrategy

from ray.util import PublicAPI

from ray_lightning import RayStrategy


# C3 linearization of parent classes will do breadth first since both
# RayStrategy and DDPSpawnShardedStrategy share
# a common parent of DDPSpawnStrategy
@PublicAPI(stability="beta")
class RayShardedStrategy(RayStrategy, DDPSpawnShardedStrategy):
    strategy_name = "ddp_sharded_ray"
