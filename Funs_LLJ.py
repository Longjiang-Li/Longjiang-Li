
import torch
import torch.nn as nn
import numpy as np
import os
import random
from skorch import NeuralNetBinaryClassifier
from sklearn import metrics


torch.manual_seed(1)


def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def seed_torch(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


seed_torch(1029)


def my_BCEWithLogitsLoss(y_pred, y_true):
    mycritical = torch.nn.BCEWithLogitsLoss()
    return mycritical(y_pred.cuda(), y_true.cuda())


def my_BCELoss(y_pred, y_true):
    mycritical = torch.nn.BCELoss()
    return mycritical(y_pred.cuda(), y_true.cuda())


class PhysinformedNet(NeuralNetBinaryClassifier):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if data != None:
            self.T_tr, self.X_tr, self.Y_tr = data.T_tr, data.X_tr.cuda(), data.Y_tr.cuda()
            self.T_val, self.X_val, self.Y_val = data.T_val, data.X_val.cuda(), data.Y_val.cuda()

    def get_loss(self, y_pred, y_true, X, training=False):
        #loss = super().get_loss(y_pred, y_true, X=X, training=training)

        loss = my_BCEWithLogitsLoss(y_pred, y_true)

        #loss = my_BCELoss(y_pred, y_true)

        return loss
