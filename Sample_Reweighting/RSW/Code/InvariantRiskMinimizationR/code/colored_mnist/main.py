# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import random

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=43)
parser.add_argument('--steps', type=int, default=4001)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

def setup_seed (seed) :
    torch.manual_seed (seed)
    torch.cuda.manual_seed_all (seed)
    np.random.seed (seed)
    random.seed (seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed (flags.seed)

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
  print("Restart", restart)

  # Load MNIST, make train/val splits, and shuffle train set examples

  mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
  mnist_train = (mnist.data[:50000], mnist.targets[:50000])
  mnist_val = (mnist.data[50000:], mnist.targets[50000:])

  rng_state = np.random.get_state()
  np.random.shuffle(mnist_train[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_train[1].numpy())

  # Build environments

  def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.).cuda(),
      'labels': labels[:, None].cuda()
    }


  envs_ = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
  ]
  
  envs = [
         {'images': torch.cat ((envs_ [0]['images'], envs_ [1]['images'])), 'labels' : torch.cat ((envs_ [0]['labels'], envs_ [1]['labels']))},
         {'images': torch.cat ((envs_ [0]['images'], envs_ [1]['images'])), 'labels' : torch.cat ((envs_ [0]['labels'], envs_ [1]['labels']))},
         envs_ [2]
  ]
  
  # Define and instantiate the model

  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        self.lin1 = nn.Linear(14 * 14, flags.hidden_dim)
      else:
        self.lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
      self.lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      self.lin3 = nn.Linear(flags.hidden_dim, 1)
      for lin in [self.lin1, self.lin2, self.lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(self.lin1, nn.ReLU(True), self.lin2, nn.ReLU(True), self.lin3)
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      out = self._main(out)
      return out
    def feature (self, input) :
        relu = nn.ReLU(True)
        return self.lin2 (relu (self.lin1 (input.view (input.shape [0], 2 * 14 * 14))))

  class LinearRandomWeight (nn.Module) :
    def __init__ (self, dim = 2 * 14 * 14) :
        super (LinearRandomWeight, self).__init__ ()
        self.linear = nn.Linear (dim, dim, bias=True)
        self.relu = nn.Sigmoid ()
        self.linear_ = nn.Linear (dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid ()
        self.weight_init ()

    def weight_init (self) :
        torch.nn.init.uniform_ (self.linear.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_ (self.linear_.weight, a=-1.0, b=1.0)

    def forward (self, x) :
        weight = self.sigmoid (self.linear_ (self.relu (self.linear (x))))
        weight = weight.detach ().clone ().view (1, -1).clamp_ (0.1, 1.)
        return weight / weight.mean ()

  mlp = MLP().cuda()
  weighter = LinearRandomWeight (flags.hidden_dim).cuda () 

  # Define loss function helpers

  def mean_nll(logits, y, w):
    return nn.functional.binary_cross_entropy_with_logits(logits, y, w)

  def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

  def penalty(logits, y, w):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y, w)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

  # Train loop

  def pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

  optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

  pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

  W = {}

  train_value_max = 0.
  test_value = 0.

  for step in range(flags.steps):
    X = mlp.feature (envs [0]['images']).detach ().clone ()#.cpu ().numpy ()
        
    for envs_count in range (5) :
        W [envs_count] = weighter (X).view (-1, 1)
        W [envs_count] = W [envs_count] / W[envs_count].mean ()
    list_nll = []
    list_acc = []
    list_penalty = []

    for envs_count in range (5) :
      logits = mlp (envs [0]['images'])
      list_nll.append (mean_nll (logits, envs [0]['labels'], W [envs_count]))
      list_acc.append (mean_accuracy (logits, envs [0]['labels']))
      list_penalty.append (penalty (logits, envs [0]['labels'], W [envs_count]))
        

    train_nll = torch.stack(list_nll).mean()
    train_acc = torch.stack(list_acc).mean()
    train_penalty = torch.stack(list_penalty).mean()

    weight_norm = torch.tensor(0.).cuda()
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss = train_nll.clone()
    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight 
        if step >= flags.penalty_anneal_iters else 1.0)
    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logits = mlp (envs [2]['images'])
    test_acc = mean_accuracy (logits, envs [2]['labels'])

    if train_acc.detach().cpu().numpy() > train_value_max :
      train_value_max = train_acc.detach().cpu().numpy()
      test_value = test_acc.detach().cpu().numpy()

    if step % 100 == 0:
      pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_penalty.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )

  final_train_accs.append(train_value_max)
  final_test_accs.append(test_value)
  print('Final train acc (mean/std across restarts so far):')
  print(np.mean(final_train_accs), np.std(final_train_accs))
  print('Final test acc (mean/std across restarts so far):')
  print(np.mean(final_test_accs), np.std(final_test_accs))
