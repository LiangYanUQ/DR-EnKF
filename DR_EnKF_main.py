import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import mat73
from model.until import data_normal
from model.model import VB_Model
from EnKF.VB_solve import burger_fom
from EnKF.EnKF import observation_point,EnKF_fom,EnKF_rom,RD_EnKF

class rom:
    all_mu = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/Operator.mat")["all_mu"]    #(50,)
    Ahat = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/Operator.mat")["Ahat_H"]  
    Ahat = np.array(Ahat)  #(50,6,6)
    Fhat = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/Operator.mat")["Fhat_H"]  
    Fhat = np.array(Fhat)  #(50,6,21)
    V = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/Operator.mat")["VH"]  # (13,6)
    r = 6
    data = mat73.loadmat("/home/wang/POD/github_DREnKF/DR_EnKF/data/OpInf_Burger_mu_T5.mat",'r')["predice_err"]   
    normal = data_normal(data)
    umax = normal.umax
    umin = normal.umin


class enkf(rom):
    dt = 1e-3
    dx = 0.01
    H  = 0.5
    h  = 1e-2
    Ne = 50
    nt = 250
    Nh  = 601
    sig = 1e-3
    x_point = np.array([-3,3])
    x_ind = [250,275,300,325,350]
    x = np.linspace(-3,3,601)
    sensor=x[x_ind].reshape(-1,1)
    Reconstruct_bais = [25] + [200] * 3 + [100]
    Reconstruct_point = [1] + [100] * 3 + [100]
    encode = [2] + [128] * 3 +[25]
    kernel = 15  ##
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


model = VB_Model(enkf)
model.compile('adam', enkf.lr)
if os.path.exists(os.path.join(enkf.model_save_path, 'VB_model_1.pth')):
    model.load_model('VB_model_1')
    print('train model load')
else:
    print('model error')
    
    
tr = 0.255
T_end = 5
Ne = enkf.Ne
nt = enkf.nt
dt = enkf.dt
xh = np.linspace(-3,3,enkf.Nh)
u0 = 0.5*np.exp(-xh**2/0.5)
u_tr = burger_fom(tr,enkf.Nh,enkf.dx,dt,u0,0,T_end)
ye = observation_point(u_tr[:,nt::nt])

obser_dim = ye.shape[0]
Imax = ye.shape[1]
u0 = u0.reshape(-1,1)
uf = np.tile(u0,Ne)

sig = enkf.sig
np.random.seed(0)
Gamma = sig**2*np.eye(obser_dim)

mu_prior = np.array([0.5])
Gamma_prior = np.array([2*10**(-2)])*np.eye(1)
sample = np.random.multivariate_normal(mean = mu_prior,cov =Gamma_prior,size =Ne)
x_H = np.linspace(-3,3,13)
u0_H = 0.5*np.exp(-x_H**2/0.5)
u0_H = u0_H.reshape(-1,1)
ur = enkf.V.T@u0_H
ur = np.tile(ur,Ne)
r = enkf.r

mrun = 10
# FOM
fm_mu = np.zeros((mrun,Imax+1,1))
fm_err = np.zeros((mrun,Imax+1,1))
fm_ur = np.zeros((mrun,Imax+1,enkf.Nh))
# ROM
rm_mu = np.zeros((mrun,Imax+1,1))
rm_err = np.zeros((mrun,Imax+1,1))
rm_ur = np.zeros((mrun,Imax+1,enkf.r))
# ROM-D
rd_mu = np.zeros((mrun,Imax+1,1))
rd_err = np.zeros((mrun,Imax+1,1))
rd_ur = np.zeros((mrun,Imax+1,enkf.r))

for mm in range(mrun):
    e = sig*np.random.normal(loc=0, scale=1, size=[obser_dim,Imax])
    y = ye+e
    # FOM
    samples_mu = sample.T
    mu_mean = np.mean(samples_mu,1) 
    fm_mu[mm,0,:] = mu_mean
    mu_mean = mu_mean.reshape(-1,1)   #(l,1)
    fm_err[mm,0,:] = np.linalg.norm(mu_mean-tr)/np.linalg.norm(tr)
    nf_mu = uf
    for j in range(Imax):
        de = y[:,j].reshape(-1,1)
        t0 = j*nt*dt
        t1 = (j+1)*nt*dt
        samples_mu,nf_mu = EnKF_fom(Ne,samples_mu,nf_mu,t0,t1,de,Gamma,enkf)
        samples_mu = samples_mu.reshape(1,-1)
        mu_mean = np.mean(samples_mu,1)  # (para_dim,)
        fm_mu[mm,j+1,:] = mu_mean
        mu_mean = mu_mean.reshape(-1,1)   #(para_dim,1)
        fm_err[mm,j+1,:] = np.linalg.norm(mu_mean-tr)/np.linalg.norm(tr)
        muint = nf_mu.mean(axis = 1)
        fm_ur[mm,j,:] = muint
    ## ROM
    samples_mu = sample.T
    mu_mean = np.mean(samples_mu,1) 
    rm_mu[mm,0,:] = mu_mean
    mu_mean = mu_mean.reshape(-1,1)   #(l,1)
    rm_err[mm,0,:] = np.linalg.norm(mu_mean-tr)/np.linalg.norm(tr)
    nr_mu = ur

    for j in range(Imax):
        de = y[:,j].reshape(-1,1)
        samples_mu,nr_mu = EnKF_rom(Ne,samples_mu,nr_mu,nt,de,Gamma,enkf)
        mu_mean = np.mean(samples_mu,1)  # (para_dim,)
        rm_mu[mm,j+1,:] = mu_mean
        mu_mean = mu_mean.reshape(-1,1)   #(para_dim,1)
        rm_err[mm,j+1,:] = np.linalg.norm(mu_mean-tr)/np.linalg.norm(tr)
        muint = nr_mu.mean(axis = 1)
        rm_ur[mm,j,:] = muint

    # # ROMD
    samples_mu = sample.T
    mu_mean = np.mean(samples_mu,1) 
    rd_mu[mm,0,:] = mu_mean
    mu_mean = mu_mean.reshape(-1,1)   #(l,1)
    rd_err[mm,0,:] = np.linalg.norm(mu_mean-tr)/np.linalg.norm(tr)
    nr_mu = ur
    for j in range(Imax):
        de = y[:,j].reshape(-1,1)
        samples_mu,nr_mu = RD_EnKF(Ne,samples_mu,nr_mu,nt,de,Gamma,(j+1)*nt*dt,enkf,model)
        mu_mean = np.mean(samples_mu,1)  # (para_dim,)
        rd_mu[mm,j+1,:] = mu_mean
        mu_mean = mu_mean.reshape(-1,1)   #(para_dim,1)
        rd_err[mm,j+1,:] = np.linalg.norm(mu_mean-tr)/np.linalg.norm(tr)
        muint = nr_mu.mean(axis = 1)
        rd_ur[mm,j,:] = muint
  
fm_mean = np.mean(fm_err,0)
rm_mean = np.mean(rm_err,0)
rd_mean = np.mean(rd_err,0)

plt.figure()
plt.semilogy(fm_mean,label = 'EnKF')
plt.semilogy(rm_mean,label = 'R-EnKF')
plt.semilogy(rd_mean,label = 'RD-EnKF')
plt.title(r' relative error')
plt.legend()
plt.savefig('relative_error.pdf')