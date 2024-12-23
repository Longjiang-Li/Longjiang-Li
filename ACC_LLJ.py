import numpy as np
from sklearn.metrics import confusion_matrix
from sympy import denom

from Funs_LLJ import ROC


"""
#           预测为正    预测为负    
# 真值为正      TP          FN      P
# 真值为负      FP          TN      N
#              P'          N'

# TPR: TP/(TP + FN), 真正率（也叫灵敏度、召回率），正样本中被预测为正的比例；
# FNR: FN/(TP + FN)，漏检率，正样本中被预测为负的比例；
# FPR: FP/(FP + TN), 虚警率 (false alarm rate), 负样本中被预测为正的比例, 注意其与False alarm ratio 的区别
# TNR: TN/(FP + TN)，真负率（特效度），负样本被预测为负的比例。
# PRE: TP/(TP + FP), 精确度 (Precision), 表示被分为正例的示例中实际为正例的比例
"""

'''`
Reference: Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
'''


def MyScore_CSI(TN, FP, FN, TP):
    # Threat Score (TS) or
    # Critical Success Index (Schaefer, J. T. (1990), The Critical Success Index as an Indicator of Warning Skill, WEATHER FORECAST, 5(4), 570-575.)
    CSI = TP / (TP + FP + FN)
    return CSI


def MyScore_FAR(TN, FP, FN, TP):
    # False Alarm Ratio (Barnes, L. R., et al. (2009), CORRIGENDUM: False Alarm Rate or False Alarm Ratio? WEATHER FORECAST, 24(5), 1452-1454)
    FAR = FP / (FP + TP)
    return FAR


def MyScore_POD(TN, FP, FN, TP):
    # Probability of Detection (Schaefer, J. T. (1990), The Critical Success Index as an Indicator of Warning Skill, WEATHER FORECAST, 5(4), 570-575.)
    POD = TP / (TP + FN)
    return POD


def MyScore_F(TN, FP, FN, TP, a=1):  # Define F-Measure
    TPR = TP / (TP + FN)
    PRE = TP / (TP + FP)
    F = ((a * a + 1) * PRE * TPR) / ((a * a) * PRE + TPR)
    return F


def MyScore_ACC(TN, FP, FN, TP):  # Define Accuracy Rate
    ACC = (TP + TN) / (TP + TN + FN + FP)
    return ACC


def MyScore_ERR(TN, FP, FN, TP):  # Define Error Rate
    ERR = (FP + FN) / (FP + FN + TP + TN)
    return ERR


# Youden's Index (约登指数),
# True Skill Statisic (TSS, Charles A., Doswell III., et al. On Summary Measures of Skill in Rare Event Forecasting Based on Contingency Tables[J]. Weather and Forecasting, 1990, 5:576-585)
def MyScore_TSS(TN, FP, FN, TP):
    TPR = TP / (TP + FN)  # Recall
    TNR = TN / (TN + FP)  # Specificity (特效度)
    TSS = TPR + TNR - 1
    return TSS


def MyScore_MCC(TN, FP, FN, TP):  # Define MCC (Matthews correction coefficient)
    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    numerator = (TP * TN) - (FP * FN)
    # If any of the four sums in the demonitor is zero, the denominator can be arbitrarily set to one;
    # this results in a Matthews correlation coefficient of zero, which can be shown to be the correct limiting value
    if denominator == 0:
        denominator = 1
    MCC = numerator / np.sqrt(denominator)
    return MCC


def MyScore(y_true, y_prob):
    TN, FP, FN, TP = confusion_matrix(y_true, y_prob).ravel()
    score = MyScore_TSS(TN, FP, FN, TP)
    return score
