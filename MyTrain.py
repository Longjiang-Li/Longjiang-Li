#-*- using coding=utf-8 -*-

import torch as t
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from Print_Results1 import *
from ACC_LLJ import *
from torch.utils.data import DataLoader, Subset


def adjust_learning_rate(cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    """
    不用都考虑加一个
    cur_epoch是当前的
    """
    curEpoch_iter = curEpoch_iter + 1
    cur_epoch = cur_epoch + 1
    cur_iter = (cur_epoch-1) * perEpoch_iter + curEpoch_iter
    max_iter = max_epoch * perEpoch_iter
    lr = baselr * pow((1 - 1.0 * cur_iter / max_iter), 0.9)

    return lr


def Train(XY_tr, Net):
    epoch_begin_time=0
    epoch_end_time=0
    epochs = 500
    lr = 0.0001
    epoch_no = 0  #刚开始的轮数
    num_classes = 2
    
    lr_decay = 'poly'
    optimizer_type = 'adam'

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.3])).cuda().double()

    optimizer = optim.Adam(params=Net.parameters(), lr=lr, weight_decay=0.0005)

    Print_Results_title(filename='Training_accuracy.txt',batch_size=XY_tr.batch_size,lr=lr,lr_decay=lr_decay,optimizer_type=optimizer_type, num_classes=num_classes)
    Print_Results_title(filename='Validation_accuracy.txt',batch_size=0.3,lr=lr,lr_decay=lr_decay,optimizer_type=optimizer_type, num_classes=num_classes)

    perEpoch_iters = len(XY_tr)

    print('Epoch'.rjust(5), 'Learn_Rate'.rjust(10),
          'Loss_tr'.rjust(9), 'TP_tr'.rjust(9), 'FP_tr'.rjust(9), 'FN_tr'.rjust(9), 'TN_tr'.rjust(9), 
          'FAR_tr'.rjust(9), 'POD_tr'.rjust(9), 'ACC_tr'.rjust(9), 'TSS_tr'.rjust(9), 
          'Loss_val'.rjust(9), 'TP_val'.rjust(9), 'FP_val'.rjust(9), 'FN_val'.rjust(9), 'TN_val'.rjust(9), 
          'FAR_val'.rjust(9), 'POD_val'.rjust(9), 'ACC_val'.rjust(9), 'TSS_val'.rjust(9),
          'Time'.rjust(9))
    for epoch in range(epoch_no, epochs):
        epoch_begin_time=time.time()

        epoch_loss = 0
        TN, FP, FN, TP = 0, 0, 0, 0

        Net.train()
        for iteration,(data, label) in enumerate(XY_tr):
            
            current_lr = adjust_learning_rate(cur_epoch=epoch, max_epoch=epochs, curEpoch_iter=iteration, perEpoch_iter=perEpoch_iters, baselr=lr)
            optimizer.param_groups[0]['lr'] = current_lr

            data = data.cuda()
            label = label.long().cuda()

            optimizer.zero_grad()
            output = Net(data)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            _, preds = t.max(output, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            label=label.data.cpu().numpy().squeeze().astype(np.uint8)

            TN1, FP1, FN1, TP1 = confusion_matrix(label.flatten(), preds.flatten()).ravel()
            TN = TN + TN1
            FP = FP + FP1
            FN = FN + FN1
            TP = TP + TP1
            epoch_loss += loss / len(XY_tr)

        far = MyScore_FAR(TN, FP, FN, TP)
        pod = MyScore_POD(TN, FP, FN, TP)
        acc = MyScore_ACC(TN, FP, FN, TP)
        tss = MyScore_TSS(TN, FP, FN, TP)
        csi = MyScore_CSI(TN, FP, FN, TP)
        Print_Results_data(filename='Training_accuracy.txt', epoch=epoch, TN=TN, FP=FP, FN=FN, TP=TP, FAR=far, POD=pod, 
                           ACC=acc, TSS=tss, CSI=csi, epoch_loss=epoch_loss)

        XY_val = XY_tr.dataset
        subset_size = int(XY_val.data_dim[0] * 0.3)
        indices = np.random.choice(XY_val.data_dim[0], subset_size, replace=False)
        subset = Subset(XY_val, indices)
        XY_val = DataLoader(subset, batch_size=100, shuffle=True)
        epoch_val_loss = 0
        TN_val, FP_val, FN_val, TP_val = 0, 0, 0, 0
        Net.eval()
        with t.no_grad():
            for i,(data, label) in enumerate(XY_val):
                data = data.cuda()
                label = label.long().cuda()

                val_output = Net(data)
                epoch_val_loss1 = criterion(val_output, label)

                epoch_val_loss += epoch_val_loss1 / len(XY_val)

                _, preds = t.max(val_output, 1)
                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                label = label.data.cpu().numpy().squeeze().astype(np.uint8)
                    
                TN_val1, FP_val1, FN_val1, TP_val1 = confusion_matrix(label.flatten(), preds.flatten()).ravel()
                TN_val += TN_val1
                FP_val += FP_val1
                FN_val += FN_val1
                TP_val += TP_val1
        far_val = MyScore_FAR(TN_val, FP_val, FN_val, TP_val)
        pod_val = MyScore_POD(TN_val, FP_val, FN_val, TP_val)
        acc_val = MyScore_ACC(TN_val, FP_val, FN_val, TP_val)
        tss_val = MyScore_TSS(TN_val, FP_val, FN_val, TP_val)
        csi_val = MyScore_CSI(TN_val, FP_val, FN_val, TP_val)
        Print_Results_data(filename='Validation_accuracy.txt', epoch=epoch, TN=TN_val, FP=FP_val, FN=FN_val, TP=TP_val, FAR=far_val, POD=pod_val, 
                            ACC=acc_val, TSS=tss_val, CSI=csi_val, epoch_loss=epoch_val_loss)

        epoch_end_time=time.time()
        used_time=(epoch_end_time-epoch_begin_time)
        print('{}{:11.8f}{:10.2f}{}{}{}{}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{}{}{}{}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}sec'.
              format(str(epoch).rjust(5), current_lr,
                     epoch_loss, str(TP).rjust(10), str(FP).rjust(10), str(FN).rjust(10), str(TN).rjust(10), 
                     100 * far, 100 * pod, 100 * acc, 100 * tss, 
                     epoch_val_loss, str(TP_val).rjust(10), str(FP_val).rjust(10), str(FN_val).rjust(10), str(TN_val).rjust(10), 
                     100 * far_val, 100 * pod_val, 100 * acc_val, 100 * tss_val, 
                     used_time))
    
    return Net
