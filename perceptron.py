import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix 
notes = pd.read_csv("bank_note_data.csv")
x= notes.drop('Class', axis=1) 
y= notes['Class']
print(x) 
print(y)
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3) 
mlp = MLPClassifier(max_iter=500, activation='relu') 
mlp.fit(train_x, train_y)
prediction = mlp.predict(test_x) 
print(prediction) 
print(confusion_matrix(test_y,prediction)) 
print(classification_report(test_y,prediction))
