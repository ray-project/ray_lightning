from pytorch_lightning.plugins import DDPSpawnShardedPlugin
from pytorch_lightning.plugins.precision.sharded_native_amp import \
    ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim.grad_scaler import ShardedGradScaler

from ray.util import PublicAPI

from ray_lightning import RayPlugin


# C3 linearization of parent classes will do breadth first since both
# RayPlugin and DDPSpawnShardedPlugin share a common parent of DDPSpawnPlugin
@PublicAPI(stability="beta")
class RayShardedPlugin(RayPlugin, DDPSpawnShardedPlugin):
    def execute_remote(self, model, global_rank, queue):
        self._model = model
        # Ensure that the scaler points to the correct process group
        # which is re-initialized in a new process
        precision_plugin = self.lightning_module.trainer.accelerator\
            .precision_plugin
        if isinstance(precision_plugin, ShardedNativeMixedPrecisionPlugin):
            precision_plugin.scaler = ShardedGradScaler()
        return super().execute_remote(
            model=model, global_rank=global_rank, queue=queue)
