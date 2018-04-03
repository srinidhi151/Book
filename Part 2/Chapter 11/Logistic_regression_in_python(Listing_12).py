#import the libraries required.
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from sklearn.linear_model import LogisticRegression
import numpy as np

#define the logistic function.
def p(feature):
   global intercept,coef
   return 1/(1+exp(-(coef*feature + intercept)))

#load the dataset as a pandas dataframe.
data = pd.read_csv('data.csv')
print(data)

#Train a logistic regression model and get the model parameters.
model = LogisticRegression()
model.fit((data['days'].values).reshape(-1,1),data['requires_service'].values)
global intercept
intercept = model.intercept_[0]
global coef
coef = model.coef_[0][0]
print(intercept,coef)

#plot the logistic function curve and the data points.
X = np.arange(0,700,0.1)
Y = list(map(p,X))
plt.xlabel('Days')
plt.ylabel('Probability')
plt.plot(X,Y,'g')
colors = {0:'black',1:'orange'}
Y_pred = model.predict((data['days'].values).reshape(-1,1))
for i in range(len(data)):
      plt.scatter(data['days'][i],
                   Data['requires_service'][i],
                   color = colors[Y_pred[i]])
plt.show()
