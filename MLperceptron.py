import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.neural_network import MLPRegressor 
from sklearn import metrics
salary = pd.read_csv('Salary.csv') 
salary = salary.drop('Employee', axis=1)
salary=pd.get_dummies(salary, columns=['MBA'], drop_first=True) 
x = salary.drop('Salary', axis=1)
y = salary[['Salary']]
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.6, random_state=2) 
scaler = MinMaxScaler()
scaler.fit(train_x) 
train_x=scaler.transform(train_x) 
test_x=scaler.transform(test_x) 
print(pd.DataFrame(train_x).describe()) 
print(pd.DataFrame(test_x).describe())
nn = MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu', max_iter=10000, solver='lbfgs')
nn.fit(train_x,train_y) 
mae=metrics.mean_absolute_error(test_y, nn.predict(test_x)) 
mse=metrics.mean_squared_error(test_y, nn.predict(test_x)) 
rsq=metrics.r2_score(test_y, nn.predict(test_x)) 
print(mae,mse,rsq)