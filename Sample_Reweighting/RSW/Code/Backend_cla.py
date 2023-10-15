import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from torch.autograd import grad

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

# Feature selection part
class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.00 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        self.mu = torch.nn.Parameter(0.00 * torch.randn(self.input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self
   
 
class MpModel (nn.Module):
    def __init__(self, input_dim, output_dim=1, sigma=0.05, lam=0., alpha=1e1, hard_sum = 98.0, penalty='Ours'):
    
        super(MpModel, self).__init__()
        self.backmodel = LinearRegression(input_dim, output_dim)
        self.mseloss = nn.MSELoss()
        self.featureSelector = FeatureSelector(input_dim, sigma)
        self.reg = self.featureSelector.regularizer
        self.lam = lam
        self.mu = self.featureSelector.mu
        self.sigma = self.featureSelector.sigma
        self.alpha = alpha
        self.optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3},
                                     {'params': self.mu, 'lr': 3e-4}])
        self.penalty = penalty
        self.hard_sum = hard_sum
        self.input_dim = input_dim
        self.accumulate_mip_penalty = torch.tensor(np.zeros(10, dtype=np.float32))

    def renew(self):
        self.featureSelector.renew()
        self.mu = self.featureSelector.mu
        self.backmodel.weight_init()
        self.optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3},
                                     {'params': self.mu, 'lr': 3e-4}])


    def combine_envs(self, envs):
        X = []
        y = []
        w = []
        for env in envs:
            X.append(env[0])
            y.append(env[1])
            w.append(env[2])
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        w = torch.cat(w, dim=0)
        return X.reshape(-1, X.shape[1]), y.reshape(-1,1), w.reshape(-1, 1)

    def pretrain(self, envs, pretrain_epoch=100):

        pre_optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3}])
        X, y, w = self.combine_envs(envs)

        BCE = nn.BCEWithLogitsLoss ()

        for i in range(pretrain_epoch):
            self.optimizer.zero_grad()
            pred = self.backmodel(X)
            loss = BCE (pred, y.reshape (pred.shape))
            loss.backward()
            pre_optimizer.step()


    def single_forward(self, x, regularizer_flag = False):
        output_x = self.featureSelector(x)
        if regularizer_flag == True:
            x = output_x.clone().detach()
        else:
            x = output_x
        return self.backmodel(x)


    def single_iter_mip(self, envs):
        assert type(envs) == list
        num_envs = len(envs)
        loss_avg = 0.0
        grad_avg = 0.0
        grad_list = []
        
        for [x,y,w] in envs:
            pred = self.single_forward(x)
            BCE = nn.BCEWithLogitsLoss (weight=w)
            loss = BCE (pred, y.reshape (pred.shape))
            loss_avg += loss/num_envs
        
        for [x,y,w] in envs:
            pred = self.single_forward(x, True) 
            BCE = nn.BCEWithLogitsLoss (weight=w)
            loss = BCE (pred, y.reshape (pred.shape))
            grad_single = grad(loss, self.backmodel.parameters(), create_graph=True)[0].reshape(-1)
            grad_avg += grad_single / num_envs
            grad_list.append(grad_single)

        penalty = torch.tensor(np.zeros(self.input_dim, dtype=np.float32))
        
        for gradient in grad_list:
            penalty += (gradient - grad_avg)**2

        penalty_detach = torch.sum(penalty.reshape(self.mu.shape)*(self.mu+0.5))
        reg = torch.sum(self.reg((self.mu + 0.5) / self.sigma))
        reg = (reg-self.hard_sum)**2
        total_loss = loss_avg + self.alpha * (penalty_detach)
        total_loss = total_loss + self.lam * reg
        return total_loss, penalty_detach, self.reg((self.mu + 0.5) / self.sigma), loss_avg


    def get_gates(self):
        return pretty(self.mu+0.5)

    def get_paras(self):
        return pretty(self.backmodel.linear.weight)

    def get_reg (self) :
        return self.reg((self.mu + 0.5) / self.sigma)

    def train(self, envs, epochs):
        for epoch in range(1,epochs+1):
            self.optimizer.zero_grad()
            loss, penalty, reg, error = self.single_iter_mip(envs)
            loss.backward()
            self.optimizer.step()
        return self.mu + 0.5, reg

    def test (self, envs) :
        assert type (envs) == list
        num_envs = len (envs)
        
        for [x, y] in envs :
            pred = self.single_forward (x)
            pred = ((pred).sign () + 1).sign ()
            loss = 1 - (torch.abs (pred - y) / y.shape [0]).sum ()
        
        return loss

