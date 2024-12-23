import h5py
import numpy as np
import random
from torch.utils.data import Dataset

def Normalize(data=None,mean_X=None,std_X=None):
    if mean_X == None:
        mean_X = data.mean(axis=0)
        std_X = data.std(axis=0)

    data = (data - mean_X) / std_X
    return data

class MyData(Dataset):
    def __init__(self, filename='', X_name='', Y_name='', DT_name='', mean_X=None, std_X=None):
        self.filename  = filename

        # 矩阵的维度是相反的，需要使用np.transpose(XY, (4,3,2,1,0)) 进行转置。
        XY = h5py.File(self.filename, 'r')
        dims_X = XY[X_name].shape
        dims_Y = XY[Y_name].shape
        dims_dt= XY[DT_name].shape
        ind_x = [len(dims_X) - i - 1 for i in range(len(dims_X))]
        ind_y = [len(dims_Y) - i - 1 for i in range(len(dims_Y))]
        ind_dt= [len(dims_dt) - i - 1 for i in range(len(dims_dt))]

        self.data = np.array(np.transpose(XY[X_name], ind_x)).astype(np.float64)
        if mean_X == None:
            mean_X = self.data.mean(axis=0)
            std_X = self.data.std(axis=0)
            std_X[std_X == 0] = 1
        self.data = (self.data - mean_X) / std_X
        self.labels = np.array(np.transpose(XY[Y_name], ind_y)).squeeze().astype(np.float64)
        self.datetime = np.array(np.transpose(XY[DT_name], ind_dt)).astype(np.float64)
        self.mean_X = mean_X
        self.std_X = std_X

        self.data_dim = self.data.shape
        self.labels_dim = self.labels.shape
        self.datetime_dim = self.datetime.shape

        print_str = f'X{X_name}.shape: ('
        for i in range(len(self.data_dim)):
            print_str = print_str + str(self.data_dim[i])
            if i == len(self.data_dim) - 1:
                print_str = print_str + ')'
            else:
                print_str = print_str + ', '
            
        print(print_str)

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_part(self, rate):
        xy = self
        n = self.dim[0]
        id = list(range(0, n))
        id = random.shuffle(id)
        id = id[:int(n * rate)]
        xy.data = self.data[id, :, :, :]
        xy.labels = self.labels[id, :, :]
        xy.datetime = self.datetime[id,:]
        xy.data_dim = xy.data.shape
        xy.labels_dim = xy.labels.shape
        xy.datetime_dim = xy.datetime.shape
        return xy