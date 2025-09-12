from supervised_learning.regression import LinearRegression as LR
import numpy as np
import math
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from utils import train_test_split
import matplotlib.pyplot as plt

def mse(a,b):
    return np.mean(0.5 * (a-b)**2)

X = np.random.rand(100,1)*3
y = 4*X + 2 + np.random.rand(100,1)
X_train,X_test,y_train,y_test = train_test_split(X,y)

print(X.shape)
print(y.shape)
model = LR()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
# print(y_pred)
print(mse(y_test,y_pred))



pX = np.array([[0],[3]])
pY = model.predict(pX)

plt.scatter(X,y)
plt.plot(pX,pY,'red')
plt.show()
