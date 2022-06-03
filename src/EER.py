
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:35:34 2017
@author: Admin

https://github.com/abhijeet3922/User-Verification
-based-on-Keystroke-Dynamics/blob/master/EER.py
"""
from sklearn.metrics import roc_curve,auc
import numpy as np

def evaluateEER(user_scores, imposter_scores):
    labels = [0]*len(user_scores) + [1]*len(imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    # print('fpr','tpr','thres',fpr,tpr,thresholds)

    missrates = 1 - tpr
    farates = fpr

    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = x[0] + a * ( y[0] - x[0] )
    return eer
