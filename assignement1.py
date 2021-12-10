#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:40:55 2021

@author: mehulkc
"""

import numpy as np
# Numerical python high d array op
import matplotlib.pyplot as plt
# lib for ploting (data visualisation)
import pandas as pd
# data manipulation and data analysis
from sklearn.impute import SimpleImputer
# Machine learning library
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('iris.csv')
# import dataset
x = dataset.iloc[:, :-1].values
# 1: row (no row skipped), 2:coloum (last col skipped)
y = dataset.iloc[:, 4].values
# y result
print(x)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean',
                        verbose='0')
# simple imputer to replace missing values.

imputer = imputer.fit(x[:, 0:3])
# fit all row and 0-3 cols
x[:, 0:3] = imputer.transform(x[:, 0:3])
# to update imputed array
print("before", x)
labelencoder_x = LabelEncoder()
# to encode col
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
print("after", x)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=0)
print(x_train.shape)
# shape means no of rows and colomns
