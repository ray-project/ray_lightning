import os
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.strategies import Strategy
from pytorch_lightning import LightningModule, Callback, Trainer, \
    LightningDataModule

import torchmetrics


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.val_epoch = 0

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # Arbitrary loss to have a loss that updates the model weights
        # during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction,
                                            torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        self.layer(batch)
        loss = torch.tensor(1.0)
        self.log("val_loss", loss)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()
        self.val_epoch += 1

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            RandomDataset(32, 64), num_workers=1)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["val_epoch"] = self.val_epoch

    def on_load_checkpoint(self, checkpoint) -> None:
        self.val_epoch = checkpoint["val_epoch"]


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        acc = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)


class XORModel(LightningModule):
    def __init__(self, input_dim=2, output_dim=1):
        super(XORModel, self).__init__()
        self.save_hyperparameters()
        self.lin1 = torch.nn.Linear(input_dim, 8)
        self.lin2 = torch.nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_nb):
        x, y = batch["x"], batch["y"].unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch["x"], batch["y"].unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log("val_loss", loss, on_step=True)
        # Log a constant for test purpose
        self.log("val_bar", torch.tensor(5.678), on_step=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_val_loss", avg_loss)
        # Log a constant for test purpose
        self.log("val_foo", torch.tensor(1.234))


class XORDataModule(LightningDataModule):
    def train_dataloader(self):
        input_train = [{
            "x": torch.tensor([[0.0, 0.0]]),
            "y": torch.tensor([0])
        }, {
            "x": torch.tensor([[1.0, 1.0]]),
            "y": torch.tensor([0])
        }]
        return iter(input_train)

    def val_dataloader(self):
        input_val = [{
            "x": torch.tensor([[0.0, 1.0]]),
            "y": torch.tensor([1])
        }, {
            "x": torch.tensor([[1.0, 0.0]]),
            "y": torch.tensor([1])
        }]
        return iter(input_val)


def get_trainer(dir,
                strategy: Strategy,
                max_epochs: int = 1,
                limit_train_batches: int = 10,
                limit_val_batches: int = 10,
                callbacks: Optional[List[Callback]] = None,
                enable_checkpointing: bool = True,
                **trainer_kwargs) -> Trainer:
    """Returns a Pytorch Lightning Trainer with the provided arguments."""
    callbacks = [] if not callbacks else callbacks
    trainer = pl.Trainer(
        default_root_dir=dir,
        callbacks=callbacks,
        strategy=strategy,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=False,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs)
    return trainer


def train_test(trainer: Trainer, model: LightningModule):
    """Checks if training the provided model updates its weights."""
    initial_values = torch.tensor(
        [torch.sum(torch.abs(x)) for x in model.parameters()])
    trainer.fit(model)
    post_train_values = torch.tensor(
        [torch.sum(torch.abs(x)) for x in model.parameters()])
    assert trainer.state.finished, f"Trainer failed with {trainer.state}"
    # Check that the model is actually changed post-training.
    assert torch.norm(initial_values - post_train_values) > 0.1


def load_test(trainer: Trainer, model: LightningModule):
    """Checks if the model checkpoint can be loaded."""
    trainer.fit(model)
    trained_model = BoringModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)
    assert trained_model is not None, "loading model failed"


def predict_test(trainer: Trainer, model: LightningModule,
                 dm: LightningDataModule):
    """Checks if the trained model has high accuracy on the test set."""
    trainer.fit(model, datamodule=dm)
    model = trainer.lightning_module
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    acc = torchmetrics.Accuracy()
    for batch in test_loader:
        x, y = batch
        with torch.no_grad():
            y_hat = model(x)
        y_hat = y_hat.cpu()
        acc.update(y_hat, y)
    average_acc = acc.compute()
    assert average_acc >= 0.5, f"This model is expected to get > {0.5} in " \
                               f"test set (it got {average_acc})"
