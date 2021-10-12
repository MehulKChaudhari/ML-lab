#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:40:55 2021

@author: mehulkc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('iris.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
print(x)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean',
verbose='0')
imputer = imputer.fit(x[:, 0:3])
x[:, 0:3] = imputer.transform(x[:, 0:3])
print(x)
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=0)
print(x_train.shape)