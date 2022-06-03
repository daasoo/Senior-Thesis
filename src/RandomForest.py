"""
Created on Sat Jan 18 2020
@author: daasoo

"""


import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from EER import evaluateEER


data = pd.read_csv("data.csv")
subjects = data["subject"].unique()

for sub in subjects:
    genuine = data.loc[data.subject == sub, "H.period":"H.Return"]
    impostor = data.loc[data.subject != sub, :]

    train_genuine = genuine[:200]
    train_impostor = impostor.groupby("subject").nth([100,200,300,399]).loc[:, "H.period":"H.Return"] #5, 55, 105, 155, 205, 255, 305, 355]

    #Data for Training (genuine and impostor)
    train = train_genuine.append(train_impostor)
    train['label'] = [0 for i in range(200)] + [1 for i in range(200)]
    train_s = train.sample(frac=1, random_state=0).reset_index(drop=True)

    train_data = train_s.loc[:, "H.period":"H.Return"]
    train_label = train_s.loc[:, "label"]

    parameters = {"n_estimators":[100, 200, 300], "max_depth":[2,3,4,5,6,7,8]}

    clf = sklearn.model_selection.GridSearchCV(RandomForestClassifier(random_state=100), parameters, scoring='accuracy', cv=5, n_jobs=-1)
    clf.fit(train_data, train_label)
    clf_best=clf.best_estimator_

    #Data for Testing (genuine: 0, impostor: 1)
    test_genuine = genuine[200:]
    test_impostor = impostor.groupby("subject").head(5).loc[:, "H.period":"H.Return"]

    genuine_score = clf_best.predict_proba(test_genuine)[:,1]
    impostor_score = clf_best.predict_proba(test_impostor)[:,1]

    test_data = test_genuine.append(test_impostor)
    test_label = [0 for i in range(200)] + [1 for i in range(250)]


    print("Subject: ", sub)
    print("EER: ", evaluateEER(list(genuine_score), list(impostor_score)))
    print("Best Accuracy Score: ", clf.score(test_data, test_label))
    print("Best Parameters: ", clf.best_params_)
    print("Best Mean cv score: ", clf.best_score_)
    print("-----------------------------------------")
