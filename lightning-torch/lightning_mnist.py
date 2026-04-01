import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,random_split

from torchvision.datasets import MNIST
from torchvision import transforms
from simple_dense_net import SimpleDenseNet
#from torchmetrics import Accuracy
#from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import LightningModule, LightningDataModule, Trainer



    
class MNISTModel(LightningModule):
    def __init__(self, ):
        super().__init__()
        self.model = SimpleDenseNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.dataset = MNIST("/Users/luyuan/neulife/pyproject/deepLearning-Basic/data", train=True, download=True, transform=transforms.ToTensor())

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int)->torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        if batch_idx % 10 == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=64, shuffle=True)
  

if __name__ == "__main__":
    model = MNISTModel()
    trainer = Trainer(max_epochs=10,accelerator="gpu")
    trainer.fit(model)