import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import math
import ERM
import ERM_cla


parser = argparse.ArgumentParser ()  
parser.add_argument ('--data_path', type=str, default=None, help='Path to data files')
parser.add_argument ('--data_type', type=int, default=1, help='coff of lasso')
parser.add_argument ('--randomseed', type=int, default=100, help='')
parser.add_argument ('--lambda1', type=float, default=0.01, help='')
parser.add_argument ('--lambda2', type=float, default=0.01, help='')
parser.add_argument ('--lambda3', type=float, default=1e5, help='penalty of invariance term')
parser.add_argument ('--num_env', type=int, default=3, help='the number of envs')
parser.add_argument ('--bias', action='store_true', help='if has bias term')
opt = parser.parse_args ()

def setup_seed (seed) :
    torch.manual_seed (seed)
    torch.cuda.manual_seed_all (seed)
    np.random.seed (seed)
    random.seed (seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed (opt.randomseed)

np.set_printoptions (4, suppress=True)

if opt.data_type == 1 :
    data = np.load ('./Data/' + opt.data_path + '/' + '0.npy', allow_pickle=True).item () ['data']
elif opt.data_type == 2 :
    data = np.load ('./Data/house.npy', allow_pickle=True).item () ['data']
elif opt.data_type == 3 :
    data = np.load ('./Data/adult_raceandsex.npy', allow_pickle=True).item () ['data0']
    
X = torch.from_numpy (data [:, : data.shape [1] - 1]).type (torch.FloatTensor)
Y = torch.from_numpy (data [:, data.shape [1] - 1]).type (torch.FloatTensor).view (-1, 1)

if opt.data_type == 3 :
    A = ERM_cla.JW (X.size () [1], 0.1, 1000, 50000, True)   
    A.train ([X, Y], 8)
else :
    A = ERM.JW (X.size () [1], opt.lambda1, opt.lambda2, opt.lambda3, opt.bias)
    A.train ([X, Y], opt.num_env)

############test################

envs_test = []
I = 10
    
if opt.data_type == 2 :
    I = 5
if opt.data_type == 3 :
    I = 9

for i in range (0, I) :
    if opt.data_type == 1 :
        data = np.load ('./Data/' + opt.data_path + '/' + '0.npy', allow_pickle=True).item () ['test' + str (i)]
    elif opt.data_type == 2 :
        data = np.load ('./Data/house.npy', allow_pickle=True).item () ['test' + str (i)]
    elif opt.data_type == 3 :
        data = np.load ('./Data/adult_raceandsex.npy', allow_pickle=True).item () ['data' + str (i + 1)]
        
    X = torch.from_numpy (data [:, : data.shape [1] - 1]).type (torch.FloatTensor)
    Y = torch.from_numpy (data [:, data.shape [1] - 1]).type (torch.FloatTensor).view (-1, 1)
    
    if opt.data_type == 1 or opt.data_type == 3 :
        envs_test.append (A.test ([X, Y]).item ())
    else :
        envs_test.append (math.sqrt (A.test ([X, Y]).item ())* 367118.7031813722)

if opt.data_type == 2 or opt.data_type == 3 :
    print (np.array (envs_test))

print (np.array (envs_test).mean ())
print (np.array (envs_test).std ())
print (np.array (envs_test).max ())

