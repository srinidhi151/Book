#import statemtents
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


#Input data of X and Y
X = np.array([4,5,7,9,10]).reshape(5, 1)
y = np.array([6, 7, 10, 11, 12]).reshape(5, 1)


# Plot the input data
plt.plot(X, y, 'ro')
axes = plt.gca()
axes.set_xlim([2, 12])
axes.set_ylim([2, 15])
plt.show()


#Build Linear regression model
model = LinearRegression()
model.fit(X, y)


#prediction vector a
a = model.coef_ * X + model.intercept_

#Plot the input points with predicted model
plt.plot(X, y, 'ro', X, a)
axes = plt.gca()
axes.set_xlim([2, 12])
axes.set_ylim([2, 15])
plt.show()
