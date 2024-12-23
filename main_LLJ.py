
import os
from Funs_LLJ import *
from MyDataset import *
from MyModel2 import *
from MyTrain import Train
from ACC_LLJ import *
from torch.utils.data import DataLoader
from Print_Results1 import *
import pickle
from MyTest import *

seed_torch(1029)

filename_tr  = 'E:\\MyResearches\\MyPapers\\待投\\Tomography_Poland\\Training\\Tr.mat'
filename_te  = 'E:\\MyResearches\\MyPapers\\待投\\Tomography_Poland\\Training\\Te.mat'
filename_net = 'E:\\MyResearches\\MyPapers\\待投\\Tomography_Poland\\Training\\Poland.pth'

print('---------------------------------------------Load X, Y------------------------------------------------')
XY_tr = MyData(filename=filename_tr, X_name='X_tr', Y_name='Y_tr', DT_name='DT_tr')
XY_tr = DataLoader(XY_tr, batch_size=100, shuffle=True)

print('---------------------------------------------Training------------------------------------------------')
if os.path.exists(filename_net):
    with open(filename_net, 'rb') as f:
        net = pickle.load(f)
else:
    net = CNN_LLJ(in_channels=XY_tr.dataset.data_dim[1], out_channels=2, midle_channels = 660)
    net = Train(XY_tr, net)
    with open(filename_net, 'wb') as f:
        pickle.dump(net, f)
Test(net, XY_tr, 'TainingDataset')

print('---------------------------------------------Test------------------------------------------------')
XY_te = MyData(filename=filename_te, X_name='X_te', Y_name='Y_te', DT_name='DT_te')
XY_te = DataLoader(XY_te, batch_size=XY_te.data_dim[0], shuffle=False)
Test(net, XY_te, 'TestDataset')
