# Distributed PyTorch Lightning Training on Ray
This library adds new PyTorch Lightning plugins for distributed training using the Ray distributed computing framework.

These PyTorch Lightning Plugins on Ray enable quick and easy parallel training while still leveraging all the benefits of PyTorch Lightning and using your desired training protocol, either [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) or [Horovod](https://github.com/horovod/horovod). 

Once you add your plugin to the PyTorch Lightning Trainer, you can parallelize training to all the cores in your laptop, or across a massive multi-node, multi-GPU cluster with no additional code changes.

This library also comes with an integration with [Ray Tune](tune.io) for distributed hyperparameter tuning experiments.

## Installation
You can install the master branch of ray_lightning like so:

`pip install git+https://github.com/ray-project/ray_lightning#ray_lightning`

## PyTorch Distributed Data Parallel Plugin on Ray
The `RayPlugin` provides Distributed Data Parallel training on a Ray cluster. PyTorch DDP is used as the distributed training protocol, and Ray is used to launch and manage the training worker processes.

Here is a simplified example:

```python
import pytorch_lightning as pl
from ray_lightning import RayPlugin

# Create your PyTorch Lightning model here.
ptl_model = MNISTClassifier(...)
plugin = RayPlugin(num_workers=4, cpus_per_worker=1, use_gpu=True)

# Don't set ``gpus`` in the ``Trainer``.
# The actual number of GPUs is determined by ``num_workers``.
trainer = pl.Trainer(..., plugins=[plugin])
trainer.fit(ptl_model)
```

Because Ray is used to launch processes, instead of the same script being called multiple times, you CAN use this plugin even in cases when you cannot use the standard `DDPPlugin` such as 
- Jupyter Notebooks, Google Colab, Kaggle
- Calling `fit` or `test` multiple times in the same script

## Horovod Plugin on Ray
Or if you prefer to use Horovod as the distributed training protocol, use the `HorovodRayPlugin` instead.

```python
import pytorch_lightning as pl
from ray_lightning import HorovodRayPlugin

# Create your PyTorch Lightning model here.
ptl_model = MNISTClassifier(...)

# 2 workers, 1 CPU and 1 GPU each.
plugin = HorovodRayPlugin(num_workers=2, use_gpu=True)

# Don't set ``gpus`` in the ``Trainer``.
# The actual number of GPUs is determined by ``num_workers``.
trainer = pl.Trainer(..., plugins=[plugin])
trainer.fit(ptl_model)
```

## Model Parallel Sharded Training on Ray
The `RayShardedPlugin` integrates with [FairScale](https://github.com/facebookresearch/fairscale) to provide sharded DDP training on a Ray cluster.
With sharded training, leverage the scalability of data parallel training while drastically reducing memory usage when training large models.

```python
import pytorch_lightning as pl
from ray_lightning import RayShardedPlugin

# Create your PyTorch Lightning model here.
ptl_model = MNISTClassifier(...)
plugin = RayShardedPlugin(num_workers=4, cpus_per_worker=1, use_gpu=True)

# Don't set ``gpus`` in the ``Trainer``.
# The actual number of GPUs is determined by ``num_workers``.
trainer = pl.Trainer(..., plugins=[plugin])
trainer.fit(ptl_model)
```
See the [Pytorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#sharded-training) for more information on sharded training.

## Multi-node Distributed Training
Using the same examples above, you can run distributed training on a multi-node cluster with just 2 simple steps.
1) [Use Ray's cluster launcher](https://docs.ray.io/en/master/cluster/launcher.html) to start a Ray cluster- `ray up my_cluster_config.yaml`.
2) [Execute your Python script on the Ray cluster](https://docs.ray.io/en/master/cluster/commands.html#running-ray-scripts-on-the-cluster-ray-submit)- `ray submit my_cluster_config.yaml train.py`. This will `rsync` your training script to the head node, and execute it on the Ray cluster.

You no longer have to set environment variables or configurations and run your training script on every single node.

## Hyperparameter Tuning with Ray Tune
`ray_lightning` also integrates with Ray Tune to provide distributed hyperparameter tuning for your distributed model training. You can run multiple PyTorch Lightning training runs in parallel, each with a different hyperparameter configuration, and each training run parallelized by itself. All you have to do is move your training code to a function, pass the function to tune.run, and make sure to add the appropriate callback (Either `TuneReportCallback` or `TuneReportCheckpointCallback`) to your PyTorch Lightning Trainer.

Example using `ray_lightning` with Tune:

```python
from ray_lightning import RayPlugin
from ray_lightning.tune import TuneReportCallback

def train_mnist(config):
    
    # Create your PTL model.
    model = MNISTClassifier(config)

    # Create the Tune Reporting Callback
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    
    trainer = pl.Trainer(
        max_epochs=4,
        callbacks=callbacks,
        plugins=[RayPlugin(num_workers=4, use_gpu=False)])
    trainer.fit(model)
    
config = {
    "layer_1": tune.choice([32, 64, 128]),
    "layer_2": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
}

# Make sure to specify how many actors each training run will create via the "extra_cpu" field.
analysis = tune.run(
        train_mnist,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        resources_per_trial={
            "cpu": 1,
            "extra_cpu": 4
        },
        name="tune_mnist")
        
print("Best hyperparameters found were: ", analysis.best_config)
```
## FAQ
> RaySGD already has a [Pytorch Lightning integration](https://docs.ray.io/en/master/raysgd/raysgd_ptl.html). What's the difference between this integration and that?

The key difference is which Trainer you'll be interacting with. In this library, you will still be using Pytorch Lightning's `Trainer`. You'll be able to leverage all the features of Pytorch Lightning, and Ray is used just as a backend to handle distributed training.

With RaySGD's integration, you'll be converting your `LightningModule` to be RaySGD compatible, and will be interacting with RaySGD's `TorchTrainer`. RaySGD's `TorchTrainer` is not as feature rich nor as easy to use as Pytorch Lightning's `Trainer` (no built in support for logging, early stopping, etc.). However, it does have built in support for fault-tolerant and elastic training. If these are hard requirements for you, then RaySGD's integration with PTL might be a better option.

> I see that `RayPlugin` is based off of Pytorch Lightning's `DDPSpawnPlugin`. However, doesn't the PTL team discourage the use of spawn?

As discussed [here](https://github.com/pytorch/pytorch/issues/51688#issuecomment-773539003), using a spawn approach instead of launch is not all that detrimental. The original factors for discouraging spawn were:
1. not being able to use 'spawn' in a Jupyter or Colab notebook, and 
2. not being able to use multiple workers for data loading. 

Neither of these should be an issue with the `RayPlugin` due to Ray's serialization mechanisms. The only thing to keep in mind is that when using this plugin, your model does have to be serializable/pickleable.
