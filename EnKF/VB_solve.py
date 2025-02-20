import os
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
# from EnKF_until import get_xsq
import scipy
" burger equation solve FOM"
def burger_fom(v,N,dx,dt,u0,T0,T_end):
    """ 
    u0: N*1
    % dx = 1/(N-1);
    % x = -3:dx:3;
    """
    k = int((T_end-T0)/dt)
    A,F = getBurgersMatrices(N,dx,v)
    x = np.empty((N,k+1))
    x[:,0] = u0.copy()
    D = scipy.sparse.eye(N)-dt*A
    # D = D.tolil()
    D = D.tocsc()
    D[0,0:2] = [1,0]
    D[-1,N-2:] = [0,1]
    
    lu = scipy.sparse.linalg.splu(D)
    for j in range(1,k+1):
        ss = get_xsq(x[:,j-1])
        x[:,j] = lu.solve(-dt*F@ss+ x[:,j-1])
        # x[:,j] = np.linalg.solve(D,-dt*F@ss+ x[:,j-1])
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
def getBurgersMatrices(N,dx,mu):
    A = mu*gallery(N,1,-2,1)/(dx**2)
    A[0,:2] = [0,1]
    A[-1,N-2:] = [0,1]
    A = scipy.sparse.csr_matrix(A)
    ii = np.reshape(np.repeat(np.arange(1,N-1),2),(2*N-4,1))
    m = np.arange(2,N)
    mi = N*(N+1)//2 - (N-m)*(N-m+1)//2 - (N-m)        
    mm = N*(N+1)//2 - (N-m)*(N-m+1)//2 - (N-m) - (N-(m-2))
    jp = mi 
    jm = mm 
    pp = np.vstack((jp, jm))
    jj = np.reshape(pp.T,(2*N-4,1))
    p = np.vstack((np.ones(N-2), -np.ones(N-2)))
    vv = np.reshape(p.T,(2*N-4,1))/(2*dx)
    F = scipy.sparse.coo_matrix((vv.flatten(),(ii.flatten(),jj.flatten())),shape=(N,N*(N+1)//2))
    return A,F

def gallery(N,lower,main,upper):
    mat = np.zeros((N,N))
    
    np.fill_diagonal(mat,main)
    
    np.fill_diagonal(mat[1:],lower)
    
    np.fill_diagonal(mat[:,1:],upper)
    return mat