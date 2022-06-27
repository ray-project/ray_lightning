from pytorch_lightning.strategies import DDPSpawnShardedStrategy

from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim.grad_scaler import ShardedGradScaler

from ray.util import PublicAPI

from ray_lightning import RayStrategy

from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.utilities.imports import _FAIRSCALE_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded
else:
    OSS = ShardedDataParallel = object


# C3 linearization of parent classes will do breadth first since both
# RayStrategy and DDPSpawnShardedStrategy share a common parent of DDPSpawnStrategy
@PublicAPI(stability="beta")
class RayShardedStrategy(RayStrategy, DDPSpawnShardedStrategy):
    strategy_name = "ddp_sharded_ray"
