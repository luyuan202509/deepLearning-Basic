import os,sys

from torch.nn.modules import loss 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset,random_split

from torchmetrics import MaxMetric,MeanMetric
from torchmetrics.classification import Accuracy  # 修改这一行
from pytorch_lightning import LightningModule, LightningDataModule, Trainer

from simple_dense_net import SimpleDenseNet


 
class MNISTModel(LightningModule):
    def __init__(self, ):
        super().__init__()
        self.model = SimpleDenseNet()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # metric object for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc   = Accuracy(task="multiclass", num_classes=10)
        self.test_acc  = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss accross batch 
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for best metric to track
        self.val_acc_best = MaxMetric()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)


    def on_train_start(self) -> None:
        self.val_acc_best.reset()
        self.train_loss.reset()
        self.train_acc.reset()
    
    def model_step(self,batch):
        x,y = batch 
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y


    
    def training_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int)->torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:int)->torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def on_validation_epoch_end(self) -> None:
       acc = self.val_acc.compute()
       self.val_acc_best(acc)
       self.log("val/acc_best", self.val_acc_best, on_epoch=True, prog_bar=True, logger=True)   

    def test_step(self,batch,batch_idx:int):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def on_test_epoch_end(self) -> None:
        pass 
    def setpup(self, stage: str = None):
        pass 
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=64, shuffle=True)
  



class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data/"):
        super().__init__()
        self.data_dir = data_dir
    
    def prepare_data(self):
        pass 

    def setup(self, stage: str = None):
        dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.data_train,self.data_val,self.data_test = \
            random_split(dataset = dataset, lengths=[0.8,0.1,0.1],generator = torch.Generator()) 
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=64, shuffle=True) 
    
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=64, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=64, shuffle=False)



if __name__ == "__main__":
    model = MNISTModel()
    datamodule = MNISTDataModule("")
    trainer = Trainer(max_epochs=2,accelerator = 'gpu')
    trainer.fit(model,datamodule)
    trainer.test(model,datamodule)