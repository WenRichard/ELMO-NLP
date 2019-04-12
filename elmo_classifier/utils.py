# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 18:42
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : utils.py
# @Software: PyCharm

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


# 定义性能指标函数
def mean(item):
    return sum(item) / len(item)


def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)