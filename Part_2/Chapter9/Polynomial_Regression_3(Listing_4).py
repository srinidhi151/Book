#import staments
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#new data from (11,16)
X = np.arange(1, 16)
y = np.append(y, [24, 23, 22, 26, 22])

#Plot the data
plt.plot(X, y, 'ro')
plt.show()

#Simple linear regression model to find the best fit
X = np.arange(1, 16).reshape(15, 1)
model = LinearRegression()
model.fit(X[:10], y[:10])
model.score(X[10:], y[10:])

#Plot the linear regression model with new data
a = np.dot(X, model.coef_.transpose()) + model.intercept_
plt.plot(X, y, 'ro', X, a)
plt.show()

#Polynomial linear regression model to find the best fit
X = np.arange(1, 16).reshape(15, 1)
X = np.c_[X, X**2]
x = np.arange(1, 16, 0.1)
x = np.c_[x, x**2]
model.fit(X[:10], y[:10])
model.score(X[10:], y[10:])

#Plot the polynomial regression model with new data
a = np.dot(x, model.coef_.transpose()) + model.intercept_
plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()

#Polynomial linear regression model X**9 to find the best fit
X = np.arange(1, 16).reshape(15, 1)
X = np.c_[X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9]
x = np.arange(1, 16, 0.1)
x = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]
model.fit(X[:10], y[:10])
model.score(X[10:], y[10:])

#Plot the polynomial regression model X**9 with new data
a = np.dot(x, model.coef_.transpose()) + model.intercept_
plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
