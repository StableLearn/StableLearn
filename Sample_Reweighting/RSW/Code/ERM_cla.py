import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import Backend_cla
import torch.nn.functional as F


class LinearRandomWeight (nn.Module) :
    def __init__ (self, dim) :
        super (LinearRandomWeight, self).__init__ ()
        self.linear = nn.Linear (dim, 1, bias=False)
        self.sigmoid = nn.Tanh ()
        self.weight_init ()

    def weight_init (self) :
        torch.nn.init.xavier_uniform_ (self.linear.weight, gain=10.0)

    def forward (self, x) :
        weight =  self.sigmoid (self.linear (x)) + 1
        weight = weight.detach ().clone ().view (1, -1).clamp_ (0.01, 2.)
        return weight / weight.sum ()

np.set_printoptions (4, suppress=True)


class JW () :
    def __init__ (self, dim, Sigma, Lam, Alpha, bias) :
        super (JW, self).__init__ ()
        self.mip = Backend_cla.MpModel (input_dim=dim, sigma=Sigma, lam=Lam, alpha=Alpha, hard_sum=dim)
        self.wer = LinearRandomWeight (dim)
        self.erm = None
        self.mar = None

    def train (self, data, envsn) :
        X = data [0]
        Y = data [1]

        W0 = torch.ones (1, X.size () [0]).reshape (1, -1)
        W0 = W0 / W0.sum ()

        envs = [] 
        envs.append ([X, Y, W0])
        envs.append ([X, Y, W0])

        self.mip.renew ()
        self.mip.pretrain (envs, 1000)

        self.wer = LinearRandomWeight (X.size () [1])
        W = {}


        for i in range (3000) :
            envs = []
            for j in range (envsn) :
                self.wer.weight_init () 
                envs.append ([X, Y, self.wer (X).reshape (-1, 1)])
            ans = self.mip.train (envs, 1)
    
    def test (self, data) :
        return self.mip.test ([data])
