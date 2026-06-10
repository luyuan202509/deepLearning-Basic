# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hands-on deep learning learning repository following the book "Deep Learning from Scratch" (ゼロから作るDeep Learning) by Saito Yasui. Code implements neural network components from scratch in NumPy and progressively introduces PyTorch and PyTorch Lightning.

**Current branch:** `pytorch-learn` — transitioning from pure NumPy to PyTorch-based learning.

Note: The most complete branch is `torch-lightning`, which contains all units (3–7), CNN, and Lightning training scripts. If looking for reference implementations, check that branch.

## How to Run

Scripts are run directly — no build system, no test runner:

```bash
python unit4/two_layer_net.py
python unit4/train_neuralnet.py
python lightning-torch/pytorch_minist.py
python day01/tensors.py
```

All files handle their own import paths via the pattern:
```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```
Scripts can be invoked from the repo root or from their own directory.

## Architecture

### Two implementation tracks

1. **Pure NumPy (from scratch):** `common/` library contains hand-written implementations of every neural network component — layers, activation functions, loss functions, optimizers, training loop. Unit scripts under `unit3/` through `unit7/` exercise these.

2. **PyTorch / PyTorch Lightning:** `lightning-torch/` contains PyTorch (`nn.Module`) and PyTorch Lightning (`LightningModule`) equivalents. `day01/` starts PyTorch tensor basics.

### `common/` — the shared NumPy library

Core building blocks, all hand-implemented:

| File | Purpose |
|---|---|
| `functions.py` | Activations (sigmoid, softmax, step), losses (MSE, cross-entropy), numerical gradient (1D + multi-dim), gradient descent |
| `layers.py` | Layer classes with `forward()` / `backward()`: `Relu`, `Sigmoid`, `Affine`, `SoftmaxWithLoss`, `Dropout`, `BatchNormalization` |
| `optimizer.py` | `SGD`, `Momentum`, `AdaGrad`, `Adam`, `RMSprop`, `Nesterov` — all expose `update(params, grads)` |
| `multi_layer_net.py` | Configurable fully-connected net with OrderedDict layers, He/Xavier init, weight decay. Exposes `predict()`, `loss()`, `accuracy()`, `gradient()` (backprop) and `numerical_gradient()` |
| `multi_layer_net_extend.py` | Extended version adding Dropout and BatchNorm support |
| `trainer.py` | `Trainer` class wrapping epoch loops, mini-batch sampling, accuracy tracking. Accepts optimizer name as string (`'sgd'`, `'adam'`, etc.) |
| `util.py` | `smooth_curve()`, `shuffle_dataset()`, `im2col()`/`col2im()` (for convolution) |

### Unit progression (book chapters)

- **Unit 3** — Forward propagation: sigmoid/softmax networks, MNIST inference with pretrained weights
- **Unit 4** — Training: loss functions, numerical gradient, mini-batch SGD training on MNIST
- **Unit 5** — Backpropagation: computational graph layers (`MulLayer`, `AddLayer`), backprop-based gradient (replacing numerical gradient), gradient checking
- **Unit 6** — Optimization techniques: optimizer comparison (SGD/Momentum/AdaGrad/Adam), weight initialization (Xavier/He), BatchNorm, Dropout, weight decay for overfitting
- **Unit 7** — CNNs: `im2col`-based Conv/Pool layers, `SimpleConvNet` training on MNIST

### `dataset/` — MNIST loader

Auto-downloads from ossci-datasets S3 mirror, caches as pickle. `load_mnist(normalize=True, flatten=True, one_hot_label=False)` returns `((train_img, train_label), (test_img, test_label))`.

## Key Patterns

- **Layer interface:** Every layer has `forward(x)` → output and `backward(dout)` → dx. Affine layers store `.dW` and `.db` after backward.
- **Parameter dicts:** Networks store weights/biases in a `self.params` dict (`params['W1']`, `params['b1']`, …). Gradients are returned as a matching dict from `gradient()`.
- **Optimizer interface:** `optimizer.update(params, grads)` mutates `params` in-place.
- **Trainer pattern:** `Trainer(network, x_train, t_train, x_test, t_test, epochs, mini_batch_size, optimizer='sgd', optimizer_param={'lr': 0.01})`, then call `trainer.train()`.

## Dependencies

Inferred from code (no requirements.txt): `numpy`, `matplotlib`, `Pillow`, `torch`, `pytorch_lightning`, `torchvision`, `torchmetrics`. Python 3.10.
