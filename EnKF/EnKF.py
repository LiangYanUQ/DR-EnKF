import numpy as np
from EnKF.VB_solve import burger_fom
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator



def EnKF_fom(Ne,samples_mu,nf_mu,t0,t1,de,Gamma,config):
    mu_mean = np.mean(samples_mu,1)  
    mu_mean = mu_mean.reshape(-1,1)   
    mu_error = samples_mu - mu_mean
    u_mu = fom_solve(samples_mu,nf_mu,Ne,config.Nh,t0,t1,config.dx,config.dt)
    sh = observation_point(u_mu)
    sh_mean = sh.mean(axis = 1)  
    sh_mean = sh_mean.reshape(-1,1)   
    sh_error = sh - sh_mean
    muint = u_mu.mean(axis = 1)
    muint = muint.reshape(-1,1)
    nr_error = u_mu - muint
    cov_u_y = 1/(Ne-1)*np.einsum('ij, kj->ik',nr_error, sh_error)
    cov_alpha_u = 1/(Ne-1)*np.einsum('ij, kj->ik', mu_error,sh_error )
    cov_u = 1/(Ne-1)*np.einsum('ik, jk->ij', sh_error,sh_error )
    errorH_data = (de-sh_mean).T@(de-sh_mean) 
    if errorH_data >1e+10:
        print('error for errorH_data!!!')
    cov_gamma = np.linalg.inv(Gamma + cov_u)
    ensem_cov = np.dot(cov_alpha_u ,cov_gamma) 
    uy_cov    = np.dot(cov_u_y ,cov_gamma) 
    mu_pre = samples_mu + np.dot(ensem_cov,(de-sh))
    ur_pre = u_mu + np.dot(uy_cov,(de-sh))
    return mu_pre,ur_pre

def fom_solve(new_mu,u0,Ne,N,t0,t1,dx,dt):
    solution = np.zeros((N,Ne))
    for i in range(Ne):
        x0 = u0[:,i]
        v = new_mu[:,i]
        x = burger_fom(v,N,dx,dt,x0,t0,t1)
        solution[:,i] = x[:,-1].copy()
    return solution


def observation_point(u):
    x_ind = [250,275,300,325,350]
    x = np.linspace(-3,3,601)
    point = x[x_ind]
    select_u = u[x_ind,:]
    return select_u
def rom_operator(Ahat,Fhat,all_mu,new_mu):
    if Ahat.shape[0] != Fhat.shape[0]:
        print('function rom operator interp error!!!')
    n_1 = Ahat.shape[1]
    n_2 = Ahat.shape[2]
    m1  = Fhat.shape[1]
    m2  = Fhat.shape[2]
    N = new_mu.shape[1]
    A = np.zeros((N,n_1,n_2))
    B = np.zeros((N,m1,m2))
    for i in range(n_1):
        for j in range(n_2):
            u = Ahat[:,i,j]
            interp_a = PchipInterpolator(all_mu,u )
            A[:,i,j] = interp_a(new_mu)
        if m1==n_1 and m2==1:
            v = Fhat[:,i,:]
            interp_f = PchipInterpolator(all_mu,v )
            B[:,i,:] = interp_f(new_mu)
        if m1==n_1 and m2>1:
            for k in range(m2):
                v = Fhat[:,i,k]
                interp_f = PchipInterpolator(all_mu,v )
                B[:,i,k] = interp_f(new_mu)
    return A,B
def implicit_euler_ACU( x0, A,B,k,dt):
   
    # Check and store dimensions.
    n = len(x0)
    assert A.shape == (n,n)
    # Solve the problem at each time step.
    x = np.empty((n,k+1))
    x[:,0] = x0.copy()
    D = np.eye(n)-dt*A
    for j in range(1,k+1):
        ss = get_xsq(x[:,j-1])
        x[:,j] = np.linalg.solve(D,dt*B@ss+ x[:,j-1])
    return x
    
def get_xsq(x):
    """
    Parameters
    ----------
    x : (n,)

    """
    n =  x.shape[0]
    y = []
    for i in range(n):
        u = np.repeat(x[i],n-i)*x[i:]
        y.append(u)
    result =np.concatenate(y)
    return result
    
def solution_interp(u,H,h,x):
    xh = np.arange(x[0],x[1]+h,h)
    xH = np.arange(x[0],x[1]+H,H)
    n = u.shape[1]
    U = np.zeros((len(xh),n))
    for i in range(n):
        f = interp1d(xH,u[:,i],kind='cubic')
        U[:,i] = f(xh)
    return U
def rom_solve(Ahat,Fhat,all_mu,new_mu,u0,Ne,n,nt,dt):
    A,B = rom_operator(Ahat,Fhat,all_mu,new_mu)
    solution = np.zeros((n,Ne))
    for i in range(Ne):
        A_mu = A[i,:,:]
        B_mu = B[i,:,:]
        x0 = u0[:,i]
        x = implicit_euler_ACU( x0, A_mu,B_mu,nt,dt)
        solution[:,i] = x[:,-1].copy()
    return solution

def EnKF_rom(Ne,samples_mu,nr_mu,nt,de,Gamma,config):
    mu_mean = np.mean(samples_mu,1)  
    mu_mean = mu_mean.reshape(-1,1)    
    mu_error = samples_mu - mu_mean
    nr_mu = rom_solve(config.Ahat,config.Fhat,config.all_mu,samples_mu,nr_mu,Ne,config.r,nt,config.dt)
    u_H = config.V@ nr_mu
    u_mu = solution_interp(u_H,config.H,config.h,config.x_point)
    sh = observation_point(u_mu)

    sh_mean = sh.mean(axis = 1) 
    sh_mean = sh_mean.reshape(-1,1)  
    sh_error = sh - sh_mean
    
    muint = nr_mu.mean(axis = 1)
    muint = muint.reshape(-1,1)
    nr_error = nr_mu - muint
    cov_u_y = 1/(Ne-1)*np.einsum('ij, kj->ik',nr_error, sh_error)
    cov_alpha_u = 1/(Ne-1)*np.einsum('ij, kj->ik', mu_error,sh_error )
    cov_u = 1/(Ne-1)*np.einsum('ik, jk->ij', sh_error,sh_error )
    errorH_data = (de-sh_mean).T@(de-sh_mean) 
    if errorH_data >1e+10:
        print('error for errorH_data!!!')
    cov_gamma = np.linalg.inv(Gamma + cov_u)
    ensem_cov = np.dot(cov_alpha_u ,cov_gamma) 
    uy_cov    = np.dot(cov_u_y ,cov_gamma) 
    mu_pre = samples_mu + np.dot(ensem_cov,(de-sh))
    ur_pre = nr_mu + np.dot(uy_cov,(de-sh))
    
    return mu_pre,ur_pre

def RD_EnKF(Ne,samples_mu,nr_mu,nt,de,Gamma,t,config,model):
    mu_mean = np.mean(samples_mu,1)  
    mu_mean = mu_mean.reshape(-1,1)   
    mu_error = samples_mu - mu_mean
    nr_mu = rom_solve(config.Ahat,config.Fhat,config.all_mu,samples_mu,nr_mu,Ne,config.r,nt,config.dt)
    u_H = config.V@ nr_mu
    u_mu = solution_interp(u_H,config.H,config.h,config.x_point)
    sh = observation_point(u_mu)

    sensor = config.sensor
    para = para_time(samples_mu,t,Ne,1)
    _,out = model.predict(sensor,para)  #(Ne*(num_steps+1),32)
    out = out*0.5*(config.umax-config.umin)+0.5*(config.umax+config.umin)
    e = out.T
    sh = sh+e
    sh_mean = sh.mean(axis = 1)  
    sh_mean = sh_mean.reshape(-1,1)   
    sh_error = sh - sh_mean
    
    muint = nr_mu.mean(axis = 1)
    muint = muint.reshape(-1,1)
    nr_error = nr_mu - muint
    cov_u_y = 1/(Ne-1)*np.einsum('ij, kj->ik',nr_error, sh_error)
    cov_alpha_u = 1/(Ne-1)*np.einsum('ij, kj->ik', mu_error,sh_error )
    cov_u = 1/(Ne-1)*np.einsum('ik, jk->ij', sh_error,sh_error )
    errorH_data = (de-sh_mean).T@(de-sh_mean) 
    if errorH_data >1e+10:
        print('error for errorH_data!!!')
    cov_gamma = np.linalg.inv(Gamma + cov_u)
    ensem_cov = np.dot(cov_alpha_u ,cov_gamma) 
    uy_cov    = np.dot(cov_u_y ,cov_gamma) 
    mu_pre = samples_mu + np.dot(ensem_cov,(de-sh))
    ur_pre = nr_mu + np.dot(uy_cov,(de-sh))
    return mu_pre,ur_pre

def para_time(samples_mu,t,Ne,num_steps):
    """
        samples_mu : 4,Ne
        t: t1,...,tk
    """
    para = np.zeros((Ne*(num_steps),2))
    para[:,0] = np.tile(t,Ne)  
    mu = samples_mu.T
    mu = mu.repeat(num_steps,axis =0)
    para[:,1:] = mu

    return para