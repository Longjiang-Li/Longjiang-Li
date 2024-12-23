from ACC_LLJ import *
from Print_Results1 import *
import torch
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import scipy.io as sio


def Test(net, XY, filename):
    net.eval()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.2])).cuda().double()
    epoch_loss = 0
    with torch.no_grad():
        for i,(data, label) in enumerate(XY):
            data = data.cuda()
            label = label.long().cuda()

            output = net(data)
            epoch_loss = criterion(output, label)

            _, preds = torch.max(output, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            label = label.data.cpu().numpy().squeeze().astype(np.uint8)
                    
            TN, FP, FN, TP = confusion_matrix(label.flatten(), preds.flatten()).ravel()
            far = MyScore_FAR(TN, FP, FN, TP)
            pod = MyScore_POD(TN, FP, FN, TP)
            acc = MyScore_ACC(TN, FP, FN, TP)
            tss = MyScore_TSS(TN, FP, FN, TP)
            csi = MyScore_CSI(TN, FP, FN, TP)
            Print_Results_data(filename=filename + '.txt', epoch=0, TN=TN, FP=FP, FN=FN, TP=TP, FAR=far, POD=pod, 
                            ACC=acc, TSS=tss, CSI=csi, epoch_loss=epoch_loss)
            sio.savemat(filename+'.mat', {'y_preds': preds, 'y_true': label, 'X': data.cpu(), 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 
                                          'FAR': far, 'POD': pod, 'ACC': acc, 'TSS': tss, 'CSI':csi})