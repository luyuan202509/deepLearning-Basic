import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""此版本存在问题，后续优化"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet  # 复用你现有的 MLP 结构
import wandb


class MNISTLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=0.1):
        super().__init__()
        # 这里直接用你原来的 TwoLayerNet
        self.model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        self.learning_rate = learning_rate

    def forward(self, x):
        # x: (B, 784)
        return self.model.predict_scores(x)  # 如果 TwoLayerNet 里没有这个接口，可以直接调用内部前向

    def training_step(self, batch, batch_idx):
        x, t = batch  # t 是 one-hot
        # TwoLayerNet 的 loss 接口已经封装了 softmax+交叉熵
        loss = self.model.loss(x.cpu().numpy(), t.cpu().numpy())
        # 转回 torch scalar
        loss = torch.tensor(loss, device=self.device, dtype=torch.float32)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        train_acc = self.model.accuracy(x.cpu().numpy(), t.cpu().numpy())
        acc = torch.tensor(train_acc, device=self.device, dtype=torch.float32)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # 用 TwoLayerNet 自己算的梯度更新就比较绕，这里改为标准的 torch 优化器
        # 如果你坚持用 TwoLayerNet 里的手写更新，可以跳过 optimizer，在 training_step 里自己更新参数。
        params = []
        for name in ("W1", "b1", "W2", "b2"):
            p = nn.Parameter(torch.from_numpy(self.model.params[name]))
            setattr(self, name, p)
            params.append(p)

        optimizer = torch.optim.SGD(params, lr=self.learning_rate)

        # 同步回 TwoLayerNet 的 numpy 参数
        def _hook(_):
            for name in ("W1", "b1", "W2", "b2"):
                self.model.params[name] = getattr(self, name).detach().cpu().numpy()

        self.trainer.fit_loop.epoch_loop._on_epoch_end = _hook  # 简单同步示例，必要时你可以改成更干净的方式
        return optimizer


def main():
    # 1. 载入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    # 转成 torch Tensor
    x_train_t = torch.from_numpy(x_train).float()
    t_train_t = torch.from_numpy(t_train).float()
    x_test_t = torch.from_numpy(x_test).float()
    t_test_t = torch.from_numpy(t_test).float()

    train_ds = TensorDataset(x_train_t, t_train_t)
    val_ds = TensorDataset(x_test_t, t_test_t)

    batch_size = 100
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 2. 建立 LightningModule
    learning_rate = 0.1
    model = MNISTLightningModule(learning_rate=learning_rate)

    # 3. 建立 WandbLogger（类似 main.py 风格）
    wandb_logger = WandbLogger(
        log_model=True,
        entity="cloudlab",
        project="minist_demo",
        name=f"TwoLayerNet-lr={learning_rate}-bs={batch_size}",
        tags=["MLP", "MNIST"],
    )

    # 4. Trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    # 5. 训练 + 验证
    trainer.fit(model, train_loader, val_loader)

    # 6. 结束 wandb run（如果你需要手动控制的话）
    wandb.finish()


if __name__ == "__main__":
    main()