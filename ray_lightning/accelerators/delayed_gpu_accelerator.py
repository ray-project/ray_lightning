# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List

import torch

from pytorch_lightning.accelerators import Accelerator,\
    GPUAccelerator


class _GPUAccelerator(GPUAccelerator):
    """Accelerator for **delayed** GPU devices.

    adapted from:
    https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/accelerators/gpu.py#L43
    but remove `torch.cuda.set_device(root_device)` in `setup_environment`
    """

    def setup_environment(self, root_device: torch.device) -> None:
        """set up the environment for delayed devices.

        modified: remove `torch.cuda.set_device(root_device)`
        and call `torch.cuda.set_device(self.device)` at the later time
        inside the `ray_launcher` or `horovod_launcher`
        """
        Accelerator.setup_environment(self, root_device)

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        # modified: return None when no devices are available
        if devices:
            return [torch.device("cuda", i) for i in devices]
        else:
            return None

    @staticmethod
    def is_available() -> bool:
        # modified to always return True
        return True

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        # the delayed gpu accelerator is registered as `_gpu`
        # in the accelerator registry
        accelerator_registry.register(
            "_gpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )
