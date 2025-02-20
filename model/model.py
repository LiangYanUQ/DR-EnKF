import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import trange 
from timeit import default_timer
import matplotlib.pyplot as plt 
from typing import Dict

class Modified_DNN(nn.Module):
    def __init__(self, layer_size) -> None:
        super().__init__()
        input_dim = layer_size[0]
        hidden_dim = layer_size[1]
        output_dim = layer_size[-1]
        num_hidden = len(layer_size) - 2
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.UB = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.VB = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.hidden_layer = nn.ModuleList()
        for i in range(num_hidden):
            hidden_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.hidden_layer.append(hidden_layer)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Tanh())
    
    def forward(self, x):
        H = self.input_layer(x)
        U = self.UB(x)
        V = self.VB(x)
        for layer in self.hidden_layer:
            Z = layer(H)
            H = (1 - Z).mul(U).add(Z.mul(V))
        return self.output_layer(H)
            
class Reconstruct_bais(nn.Module):
    def __init__(self, layer_size, kernel_size) -> None:
        super().__init__()
        self.layer_size = layer_size.copy()
        self.kernel_size = kernel_size
        self.layer_size[-1] *= kernel_size
        self.Modified_DNN = Modified_DNN(self.layer_size)
        self.conv_layer = nn.Conv1d(1,1, kernel_size=kernel_size, stride = kernel_size, bias = False)
        
    def forward(self, x):
        output = self.Modified_DNN(x).unsqueeze(dim = 1)
        out = self.conv_layer(output).squeeze()
        return out

class Reconstruct_point(nn.Module):
    """This class implements for reconstruct point network"""
    def __init__(self, layer_size) -> None:
        super().__init__()
        self.layer_size = layer_size.copy()
        self.net = Modified_DNN(self.layer_size)
    def forward(self, x):
        return self.net(x)
    
class Encode(nn.Module):
    """This class implements Encode net"""
    def __init__(self, layer_size) -> None:
        super().__init__()
        input_dim = layer_size[0]
        hidden_dim = layer_size[1]
        output_dim = layer_size[-1]
        num_hidden = len(layer_size) - 2
        layer = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]

        for i in range(num_hidden):
            layer.append(nn.Linear(hidden_dim, hidden_dim))
            layer.append(nn.Tanh())
        layer.append(nn.Linear(hidden_dim, output_dim))
        # layer.append(nn.Tanh())
        self.encoder = nn.Sequential(*layer)
    def forward(self, x):
        return self.encoder(x)
    
def get_shuffled_dataloader(input_data,param_data,target_data):
    indices = list(range(len(input_data)))
    random.shuffle(indices)
    # num_batches = len(input_data)//batch_size
    shuffled_input_data = input_data[indices]
    shuffled_target_data = target_data[indices]
    shuffled_param_data = param_data[indices]
    # print(indices)
    return shuffled_input_data,shuffled_param_data,shuffled_target_data

class VB_Model:
    def __init__(self, config) -> None:
        self.En_net = Encode(config.encode.copy()).to(config.device)
        self.recB_net = Reconstruct_bais(config.Reconstruct_bais.copy(), config.kernel).to(config.device)
        self.recP_net =  Reconstruct_point(config.Reconstruct_point).to(config.device) 
        self.step_size = config.step_size
        self.device = config.device 
        self.dtype = config.dtype
        self.config = config 
        self.Nh = config.Nh
        self.m = config.m
        self.batch_size = config.batch_size
        self.myloss = torch.nn.MSELoss(reduction='mean')
        self.loss_log = {'epoch': [], 'res_loss': [],'test_loss':[],'error_loss':[],'param_loss':[]}


    def compile(self, optimizer, lr):
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam([
                                {'params': self.recB_net.parameters()}, 
                                {'params': self.recP_net.parameters()},
                                {'params':self.En_net.parameters()}],
                                lr = lr, betas = [0.99, 0.999], eps = 1e-8)
            self.lr_schedular = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=0.98)
        else:
            raise NotImplementedError("This optimizer is not implemented") 
        
    def train(self, train_data,train_param,target_data,point, test_data,test_param,test_target,nIter):
        pbar = trange(nIter)
        n_batches = len(train_param)//self.batch_size
        n_test = len(test_param)//self.batch_size
        input_test,param_test,out_test =  get_shuffled_dataloader(test_data,test_param,test_target)
        for epoch in pbar:
            train_num = 0
            train_step = 0
            train_loss = 0
            input_data,input_param,out_data =  get_shuffled_dataloader(train_data,train_param,target_data)
            for i in range(n_batches):
                idy = np.random.choice(self.Nh, (self.m,), replace=False)
                x = input_param[i*self.batch_size:(i*self.batch_size + self.batch_size),]
                u = input_data[i*self.batch_size:(i*self.batch_size + self.batch_size),]
                y = out_data[i*self.batch_size:(i*self.batch_size + self.batch_size),idy]
                sensors = point[idy,]
                self.optimizer.zero_grad()
                loss,loss_e,loss_param = self.data_loss(u,sensors,x,y)
                loss.backward()
                self.optimizer.step()
                train_step += loss.item()*x.size(0)*y.size(1)
                train_num = train_num + x.size(0)*y.size(1)
            self.lr_schedular.step()
            train_loss = train_step /train_num
            test_step = 0
            test_loss = 0
            test_num = 0
            test_num = 0
            with torch.no_grad():
                for j in range(n_test):
                    loss_t = 0
                    idy = np.random.choice(self.Nh, (self.m,), replace=False)
                    xx = param_test[j*self.batch_size:(j*self.batch_size + self.batch_size),]
                    uu = input_test[j*self.batch_size:(j*self.batch_size + self.batch_size),]
                    yy = out_test[j*self.batch_size:(j*self.batch_size + self.batch_size),idy]
                    pp = point[idy,]
                    loss_t,_,_ = self.data_loss(uu,pp,xx,yy)
                    test_step += loss_t.item()*xx.size(0)*yy.size(1)
                    test_num = test_num + xx.size(0)*yy.size(1)
                test_loss = test_step/test_num
           
            if epoch % 20 == 0 or epoch == 0:
                self.loss_log['epoch'].append(epoch)
                self.loss_log['res_loss'].append(train_loss)
                self.loss_log['test_loss'].append(test_loss)
                self.loss_log['error_loss'].append(loss_e.item())
                self.loss_log['param_loss'].append(loss_param.item())
                pbar.set_postfix({'loss_res': train_loss,'test_loss':test_loss,'error_loss':loss_e.item(),'param_loss':loss_param.item()})
        self.visualize_loss(self.config.title, self.loss_log, 'Training loss')

    def data_loss(self, inputs,sensors,params,outputs):
        B = self.recB_net(inputs)
        T = self.recP_net(sensors)
        f = torch.einsum('bi, ji -> bj', B, T)
        e =  self.En_net(params)
        loss_e = self.myloss(outputs.view(-1, 1),f.view(-1, 1))  
        loss_param = self.myloss(inputs,e)
        loss = loss_e + 0.1*loss_param
        return loss,loss_e,loss_param
    
    

    def predict(self, x,params):
        x = self.to_tensor(x)
        params = self.to_tensor(params)
        e =  self.En_net(params)
        B = self.recB_net(e)
        T = self.recP_net(x)
        out = torch.einsum('bi, ji -> bj', B, T)
        out = self.to_array(out)
        e = self.to_array(e)
        return e,out
    
    def to_array(self, x):
        if isinstance(x, torch.Tensor):
            return x.to('cpu').detach().numpy()
        return x
    
    def to_tensor(self, x):
        return torch.tensor(x, dtype=self.dtype, device = self.device)
    
    def visualize_loss(self, prefix: str, train_log:Dict ,title: str):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure()
        plt.yscale('log')
        for key, value in train_log.items():
            if key != 'epoch':
                plt.plot(train_log['epoch'], value, label = key)
        plt.title('Epoch:loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.img_save_path, prefix + '.pdf'))
        plt.close()
    def load_model(self, prefix):
        states = torch.load(os.path.join(self.config.model_save_path, prefix + '.pth'))
        self.En_net.load_state_dict(states['En'])
        self.recB_net.load_state_dict(states['recb'])
        self.recP_net.load_state_dict(states['recp'])
    
    
    def save_model(self, prefix):
        save_path = os.path.join(self.config.model_save_path, prefix)
        states = {'recb': self.recB_net.state_dict(), 'recp': self.recP_net.state_dict(),'En': self.En_net.state_dict()}
        torch.save(states, save_path) 