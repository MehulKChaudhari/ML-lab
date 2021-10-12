import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("data.csv")
x = data.iloc[:, :-1]
y = data.iloc[:, 5].values
encode_x = LabelEncoder()
x = x.apply(LabelEncoder().fit_transform)
print(x)
z = DecisionTreeClassifier()
z.fit(x.iloc[:, 1:5], y)
pred = z.predict([[1, 1, 0, 0]])
print("Prediction : ", pred)