from supervised_learning.regression import LinearRegression as LR
import numpy as np
import math
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from utils import train_test_split

def mse(a,b):
    return np.mean(0.5 * (a-b)**2)

X = np.random.rand(1000,1)
y = 4*X + np.random.rand()
X_train,X_test,y_train,y_test = train_test_split(X,y)


print(X.shape)
print(y.shape)
model = LR()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
# print(y_pred)
print(mse(y_test,y_pred))

print("BELOW IS GOOD MODEL!!")
print(X_test.shape)
model2 = LinearRegression(fit_intercept=False)
model2.fit(X_train,y_train)

y_pred = model2.predict(X_test)
# print(y_pred)
print(mse(y_test,y_pred))

