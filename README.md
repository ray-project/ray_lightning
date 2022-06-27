<!--$UNCOMMENT(ray-lightning)=-->

# Distributed PyTorch Lightning Training on Ray
This library adds new PyTorch Lightning strategies for distributed training using the Ray distributed computing framework.

These PyTorch Lightning strategies on Ray enable quick and easy parallel training while still leveraging all the benefits of PyTorch Lightning and using your desired training protocol, either [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) or [Horovod](https://github.com/horovod/horovod). 

Once you add your strategy to the PyTorch Lightning Trainer, you can parallelize training to all the cores in your laptop, or across a massive multi-node, multi-GPU cluster with no additional code changes.

This library also comes with an integration with <!--$UNCOMMENT{ref}`Ray Tune <tune-main>`--><!--$REMOVE-->[Ray Tune](https://tune.io)<!--$END_REMOVE--> for distributed hyperparameter tuning experiments.

<!--$REMOVE-->
# Table of Contents
- [Distributed PyTorch Lightning Training on Ray](#distributed-pytorch-lightning-training-on-ray)
- [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [PyTorch Lightning Compatibility](#pytorch-lightning-compatibility)
  - [PyTorch Distributed Data Parallel Strategy on Ray](#pytorch-distributed-data-parallel-strategy-on-ray)
  - [Multi-node Distributed Training](#multi-node-distributed-training)
  - [Multi-node Training from your Laptop](#multi-node-training-from-your-laptop)
  - [Horovod Strategy on Ray](#horovod-strategy-on-ray)
  - [Model Parallel Sharded Training on Ray](#model-parallel-sharded-training-on-ray)
  - [Hyperparameter Tuning with Ray Tune](#hyperparameter-tuning-with-ray-tune)
  - [FAQ](#faq)
<!--$END_REMOVE-->


## Installation
You can install Ray Lightning via `pip`:

`pip install ray_lightning`

Or to install master:

`pip install git+https://github.com/ray-project/ray_lightning#ray_lightning`

## PyTorch Lightning Compatibility
Here are the supported PyTorch Lightning versions:

| Ray Lightning | PyTorch Lightning |
|---|---|
| 0.1 | 1.4 |
| 0.2 | 1.5 |
| master | 1.5 |


## PyTorch Distributed Data Parallel Strategy on Ray
The `RayStrategy` provides Distributed Data Parallel training on a Ray cluster. PyTorch DDP is used as the distributed training protocol, and Ray is used to launch and manage the training worker processes.

Here is a simplified example:

```python
import pytorch_lightning as pl
from ray_lightning import RayStrategy

# Create your PyTorch Lightning model here.
ptl_model = MNISTClassifier(...)
strategy = RayStrategy(num_workers=4, num_cpus_per_worker=1, use_gpu=True)

# Don't set ``gpus`` in the ``Trainer``.
# The actual number of GPUs is determined by ``num_workers``.
trainer = pl.Trainer(..., strategy=strategy)
trainer.fit(ptl_model)
```

Because Ray is used to launch processes, instead of the same script being called multiple times, you CAN use this strategy even in cases when you cannot use the standard `DDPStrategy` such as 
- Jupyter Notebooks, Google Colab, Kaggle
- Calling `fit` or `test` multiple times in the same script

## Multi-node Distributed Training
Using the same examples above, you can run distributed training on a multi-node cluster with just a couple simple steps.

First, use Ray's <!--$UNCOMMENT{ref}`Cluster launcher <ref-cluster-quick-start>`--><!--$REMOVE-->[Cluster launcher](https://docs.ray.io/en/latest/cluster/quickstart.html)<!--$END_REMOVE--> to start a Ray cluster:

```bash
ray up my_cluster_config.yaml
```

Then, run your Ray script using one of the following options:

1. on the head node of the cluster (``python train_script.py``)
2. via ``ray job submit`` (<!--$UNCOMMENT{ref}`docs <jobs-overview>`--><!--$REMOVE-->[docs](https://docs.ray.io/en/latest/cluster/job-submission.html)<!--$END_REMOVE-->) from your laptop (``ray job submit -- python train.py``)

## Multi-node Training from your Laptop
Ray provides capabilities to run multi-node and GPU training all from your laptop through
<!--$UNCOMMENT{ref}`Ray Client <ray-client>`--><!--$REMOVE-->[Ray Client](https://docs.ray.io/en/master/cluster/ray-client.html)<!--$END_REMOVE-->

Ray's <!--$UNCOMMENT{ref}`Cluster launcher <ref-cluster-quick-start>`--><!--$REMOVE-->[Cluster launcher](https://docs.ray.io/en/latest/cluster/quickstart.html)<!--$END_REMOVE--> to setup the cluster.
Then, add this line to the beginning of your script to connect to the cluster:
```python
import ray
# replace with the appropriate host and port
ray.init("ray://<head_node_host>:10001")
```
Now you can run your training script on the laptop, but have it execute as if your laptop has all the resources of the cluster essentially providing you with an **infinite laptop**.

**Note:** When using with Ray Client, you must disable checkpointing and logging for your Trainer by setting `checkpoint_callback` and `logger` to `False`.

## Horovod Strategy on Ray
Or if you prefer to use Horovod as the distributed training protocol, use the `HorovodRayStrategy` instead.

```python
import pytorch_lightning as pl
from ray_lightning import HorovodRayStrategy

# Create your PyTorch Lightning model here.
ptl_model = MNISTClassifier(...)

# 2 workers, 1 CPU and 1 GPU each.
strategy = HorovodRayStrategy(num_workers=2, use_gpu=True)

# Don't set ``gpus`` in the ``Trainer``.
# The actual number of GPUs is determined by ``num_workers``.
trainer = pl.Trainer(..., strategy=strategy)
trainer.fit(ptl_model)
```

## Model Parallel Sharded Training on Ray
The `RayShardedStrategy` integrates with [FairScale](https://github.com/facebookresearch/fairscale) to provide sharded DDP training on a Ray cluster.
With sharded training, leverage the scalability of data parallel training while drastically reducing memory usage when training large models.

```python
import pytorch_lightning as pl
from ray_lightning import RayShardedStrategy

# Create your PyTorch Lightning model here.
ptl_model = MNISTClassifier(...)
strategy = RayShardedStrategy(num_workers=4, num_cpus_per_worker=1, use_gpu=True)

# Don't set ``gpus`` in the ``Trainer``.
# The actual number of GPUs is determined by ``num_workers``.
trainer = pl.Trainer(..., strategy=strategy)
trainer.fit(ptl_model)
```
See the [Pytorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#sharded-training) for more information on sharded training.

## Hyperparameter Tuning with Ray Tune
`ray_lightning` also integrates with Ray Tune to provide distributed hyperparameter tuning for your distributed model training. You can run multiple PyTorch Lightning training runs in parallel, each with a different hyperparameter configuration, and each training run parallelized by itself. All you have to do is move your training code to a function, pass the function to tune.run, and make sure to add the appropriate callback (Either `TuneReportCallback` or `TuneReportCheckpointCallback`) to your PyTorch Lightning Trainer.

Example using `ray_lightning` with Tune:

```python
from ray import tune

from ray_lightning import RayStrategy
from ray_lightning.examples.ray_ddp_example import MNISTClassifier
from ray_lightning.tune import TuneReportCallback, get_tune_resources

import pytorch_lightning as pl


def train_mnist(config):
    
    # Create your PTL model.
    model = MNISTClassifier(config)

    # Create the Tune Reporting Callback
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    
    trainer = pl.Trainer(
        max_epochs=4,
        callbacks=callbacks,
        strategy=[RayStrategy(num_workers=4, use_gpu=False)])
    trainer.fit(model)
    
config = {
    "layer_1": tune.choice([32, 64, 128]),
    "layer_2": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
}

# Make sure to pass in ``resources_per_trial`` using the ``get_tune_resources`` utility.
analysis = tune.run(
        train_mnist,
        metric="loss",
        mode="min",
        config=config,
        num_samples=2,
        resources_per_trial=get_tune_resources(num_workers=4),
        name="tune_mnist")
        
print("Best hyperparameters found were: ", analysis.best_config)
```
**Note:** Ray Tune requires 1 additional CPU per trial to use for the Trainable driver. So the actual number of resources each trial requires is `num_workers * num_cpus_per_worker + 1`.

## FAQ
> I see that `RayStrategy` is based off of Pytorch Lightning's `DDPSpawnStrategy`. However, doesn't the PTL team discourage the use of spawn?

As discussed [here](https://github.com/pytorch/pytorch/issues/51688#issuecomment-773539003), using a spawn approach instead of launch is not all that detrimental. The original factors for discouraging spawn were:
1. not being able to use 'spawn' in a Jupyter or Colab notebook, and 
2. not being able to use multiple workers for data loading. 

Neither of these should be an issue with the `RayStrategy` due to Ray's serialization mechanisms. The only thing to keep in mind is that when using this strategy, your model does have to be serializable/pickleable.

> Horovod installation issue: 
> ```
> Extension horovod.torch has not been built: /home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-38-x86_64-linux-gnu.so not found
> If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
>Warning! MPI libs are missing, but python applications are still avaiable.
> ```

One might fix this issue by 
```python
$ pip uninstall -y horovod
$ conda install gcc_linux-64 gxx_linux-64
$ [flags] pip install --no-cache-dir horovod
```

from [here](https://github.com/horovod/horovod/issues/656), [here](https://github.com/tlkh/ai-lab/issues/27) and [here](https://horovod.readthedocs.io/en/stable/install_include.html)

- install horovod from scratch with torch 

```python
conda create -n hd python=3.8 scipy numpy pandas -y
conda activate hd
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-11.3 /usr/local/cuda
conda install gxx_linux-64 -y
conda install cxx-compiler=1.0 -y
export TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0"
echo $TORCH_CUDA_ARCH_LIST
sudo apt-get purge -y cmake
wget -q https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
tar -zxvf cmake-3.20.2.tar.gz
cd cmake-3.20.2
./bootstrap -- -DCMAKE_USE_OPENSSL=OFF
make -j10
sudo make install
cmake --version
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export HOROVOD_NCCL_HOME=/usr/local/cuda/
export HOROVOD_NCCL_INCLUDE=/usr/local/cuda/include
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST//";8.0"/}
export HOROVOD_BUILD_CUDA_CC_LIST=${TORCH_CUDA_ARCH_LIST//";"/","}
export HOROVOD_BUILD_CUDA_CC_LIST=${HOROVOD_BUILD_CUDA_CC_LIST//"."/""}
export PATH=/usr/local/cuda/bin/:$PATH
export HOROVOD_NCCL_LIB=/usr/local/cuda/lib/
HOROVOD_NCCL_HOME=/usr/local/cuda HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1  HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_GLOO=1 pip install --no-cache-dir horovod
```
<!--$UNCOMMENT## API Reference

```{eval-rst}
.. autoclass:: ray_lightning.RayStrategy
```

```{eval-rst}
.. autoclass:: ray_lightning.HorovodRayStrategy
```

```{eval-rst}
.. autoclass:: ray_lightning.RayShardedStrategy
```


### Tune Integration
```{eval-rst}
.. autoclass:: ray_lightning.tune.TuneReportCallback
```

```{eval-rst}
.. autoclass:: ray_lightning.tune.TuneReportCheckpointCallback
```

```{eval-rst}
.. autofunction:: ray_lightning.tune.get_tune_resources
```-->
