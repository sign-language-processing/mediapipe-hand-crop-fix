import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mediapipe_crop_estimate.train_dataset import get_dataset


class SimpleMLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class DataModule(pl.LightningDataModule):
    # pylint: disable=redefined-outer-name
    def __init__(self, train_dataset, val_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dataset = get_dataset()

    train_features = dataset["train_input"].shape[-1]
    label_features = dataset["train_label"].shape[-1]

    models = {
        "center": [0, 1],
        "size": [2],
        "rotation": [3]
    }

    os.makedirs("mlp", exist_ok=True)

    for model_name, label_indices in models.items():
        train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'][:, label_indices])
        val_dataset = TensorDataset(dataset['test_input'][:10], dataset['test_label'][:, label_indices][:10])
        data_module = DataModule(train_dataset, val_dataset)

        model = SimpleMLP(input_size=train_features, hidden_size=10, output_size=len(label_indices))
        trainer = pl.Trainer(max_epochs=100, progress_bar_refresh_rate=20)
        trainer.fit(model, datamodule=data_module)

        # save model jit
        model.eval()
        example_input = dataset["train_input"][0]
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(f"mlp/{model_name}.pt")
