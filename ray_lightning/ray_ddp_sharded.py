from pytorch_lightning.strategies import DDPSpawnShardedPlugin
from pytorch_lightning.strategies.precision.sharded_native_amp import \
    ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim.grad_scaler import ShardedGradScaler

from ray.util import PublicAPI

from ray_lightning import RayStrategy


# C3 linearization of parent classes will do breadth first since both
# RayStrategy and DDPSpawnShardedPlugin share a common parent of DDPSpawnPlugin
@PublicAPI(stability="beta")
class RayShardedPlugin(RayStrategy, DDPSpawnShardedPlugin):
    def execute_remote(self, model, global_rank, queue):
        # Need to set self._model here otherwise self.lightning_module will
        # return None.
        self._model = model

        # This is copied from `DDPSpawnShardedPlugin.new_process`.
        # As of PTL 1.5, this is the only difference between
        # `DDPSpawnShardedPlugin` and `DDPSpawnPlugin`.
        precision_plugin = self.lightning_module.trainer.accelerator\
            .precision_plugin
        if isinstance(precision_plugin, ShardedNativeMixedPrecisionPlugin):
            precision_plugin.scaler = ShardedGradScaler()

        # After setting the grad scaler, we can now call the default
        # `RayStrategy.execute_remote`.
        return super().execute_remote(
            model=self._model, global_rank=global_rank, queue=queue)
