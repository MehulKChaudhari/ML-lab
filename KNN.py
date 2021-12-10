#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:40:55 2021

@author: mehulkc
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("kdata.csv")
print(data)
x = data.iloc[:,:-1].values
print(x)
y = data.iloc[:,2].values
print(y)
#classifier 1 for point [6,6]
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x,y)
y_pred = classifier.predict([[6,6]])
print("Prediction 1: ", y_pred)

#classifier 2 distance weighted classifier for point [6,3]
classifier2 = KNeighborsClassifier(n_neighbors=3, weights="distance")
classifier2.fit(x,y)
y_pred = classifier2.predict([[6,3]])
print("Prediction 2: ", y_pred)