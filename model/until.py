import numpy as np
class data_normal:
    def __init__(self,X):
        self.umax = X.max()
        self.umin = X.min()
        print("the maxium and minimum value for data normalization is {}, {}".format(self.umax,self.umin))
    def normal(self,X):
        X = (X - 0.5*(self.umax+self.umin))/(0.5*(self.umax-self.umin))
        return X
    def renormal(self,Y):
        Y = (Y - 0.5*(self.umax+self.umin))/(0.5*(self.umax-self.umin))
        return Y
    

def para_loader(para_data,t,num_para,num_t,ndim):
    # train_data: Ne,nt,ndim
    num_points = num_para*num_t
    para_res_train = np.zeros((num_points, ndim+1))
    num = 0 
    for i in range(num_para):
        para = para_data[i]
        for k in range(num_t):
            para_res_train[num,1:] = para
            para_res_train[num,0] = t[k]
            num = num + 1
    return para_res_train