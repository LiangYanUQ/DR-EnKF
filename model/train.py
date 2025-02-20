import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import mat73
import time
import random
from timeit import default_timer
from model import VB_Model
from until import data_normal,para_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/OpInf_Burger_mu_T5.mat",'r')["predice_err"]   
data_test = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/OpInf_Burger_testmu_T5.mat")["test_error"]   
para_data = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/OpInf_Burger_mu_T5.mat")["all_mu2"]   #(Ne,1)
para_test = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/OpInf_Burger_testmu_T5.mat")["test_mu"]

x = np.linspace(-3, 3, 601)
point = x.reshape(-1,1)
# train data
normal = data_normal(data)
u_data = normal.normal(data)
train_data = u_data[:,::25,:]  
train_data = np.transpose(train_data,(0,2,1))
n_dim = 25
train_data = train_data.reshape(-1,n_dim )  
U = np.transpose(u_data[:,:,:],(0,2,1))
U = U.reshape(-1,601)
t = np.linspace(0,5, 51)
num_para = para_data.shape[0]
num_t = len(t)
para_dim = 1
train_param = para_loader(para_data,t,num_para,num_t,para_dim)

# test data
u_data_test = normal.normal(data_test[:,:,::50])
test_data = u_data_test[:,::25,:]  
test_data = np.transpose(test_data,(0,2,1))
test_data = test_data.reshape(-1,n_dim ) 
test_U = np.transpose(u_data_test[:,:,:],(0,2,1))
test_U = test_U.reshape(-1,601)
t_test = np.linspace(0,5, 101)
num_para_test = para_test.shape[0]
num_test = len(t_test)
test_param = para_loader(para_test,t_test,num_para_test,num_test,para_dim)


p_train = torch.from_numpy(train_param).float().to(device)
U_train = torch.from_numpy(U).float().to(device)
d_train = torch.from_numpy(train_data).float().to(device)

U_test = torch.from_numpy(test_U).float().to(device)
p_test = torch.from_numpy(test_param).float().to(device)
d_test = torch.from_numpy(test_data).float().to(device)
sensors = torch.from_numpy(point).float().to(device)

class config:
    Reconstruct_bais = [25] + [200] * 3 + [100]
    Reconstruct_point = [1] + [100] * 3 + [100]
    encode = [2] + [128] * 3 +[25]
    kernel = 15  ##
    Nh = 601
    m  = 100
    lr = 1e-3
    step_size = 10000
    batch_size = 200
    num_epoch = 20000
    title = 'VB_model'
    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_save_path = '/home/wang/POD/github_DREnKF/DR_EnKF/model/img'
    model_save_path = '/home/wang/POD/github_DREnKF/DR_EnKF/model/net'

if not os.path.exists(config.img_save_path):
    os.makedirs(config.img_save_path)
if not os.path.exists(config.model_save_path):
    os.makedirs(config.model_save_path)
    
model = VB_Model(config)
model.compile('adam', config.lr)
if os.path.exists(os.path.join(config.model_save_path, 'VB_model_1.pth')):
    model.load_model('VB_model_1')
else:
    model.train(d_train,p_train,U_train,sensors,d_test,p_test,U_test, config.num_epoch)
    model.save_model('VB_model_1.pth')
  