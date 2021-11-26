from sklearn.datasets import make_blobs 
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


x, y = make_blobs(n_samples=200, centers=2, cluster_std=0.6, random_state=0) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=10) 
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap="viridis")
plt.show()
classifier = SVC(kernel='linear') 
classifier.fit(x_train, y_train)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap="viridis") 
axis = plt.gca()
xlim = axis.get_xlim()
axis.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap="winter", marker='s') 
w= classifier.coef_[0]

a = -w[0]/ w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a* xx - (classifier.intercept_[0]/w[1]) 
plt.plot(xx, yy)
plt.show()
pred_y = classifier.predict(x_test) 
print(pred_y) 
print(confusion_matrix(y_test, pred_y))