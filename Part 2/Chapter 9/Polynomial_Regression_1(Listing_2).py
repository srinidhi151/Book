#import staments
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


#Data inputs
X = np.arange(1, 11).reshape(10, 1)
y = np.array([7, 8, 7, 13, 16, 15, 19, 23, 18, 21]).reshape(10, 1)

#Plot the data points
plt.plot(X, y, 'ro')
plt.show()


#Build the linear regression model

model = LinearRegression()
model.fit(X, y)

#prediction vector a
a = model.coef_ * X + model.intercept_


#Plot the fitted model
plt.plot(X, y, 'ro', X, a)
axes = plt.gca()
axes.set_xlim([0, 30])
axes.set_ylim([0, 30])
plt.show()

print('Linear regression score',model.score(X, y))

#Polynomial features to improve the accuracy
X = np.arange(1, 11)
X = np.c_[X, X**2]
x = np.arange(1, 11, 0.1)
x = np.c_[x, x**2]

#Build the linear regression model
model.fit(X, y)
a = np.dot(x, model.coef_.transpose()) + model.intercept_

#Plot the fitted model
plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
print('X^2 model score',model.score(X, y))
