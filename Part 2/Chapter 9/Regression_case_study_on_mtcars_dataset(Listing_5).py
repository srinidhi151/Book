import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as split

data = pd.read_csv('mtcars.csv')
attributes = ['mpg', 'disp', 'hp', 'wt']
data = data[attributes]
train, test  = split(data, test_size=0.25)
print(train, test)

X = np.array(train[['disp', 'hp', 'wt']])
Y = np.array(train['mpg'])


model = LR()
model.fit(X,Y)
coeffcients = list(model.coef_)
intercept = model.intercept_
print("actual value \t predicted value")
test = np.array(test)
for item in test:
    #print(item)
    print(item[0],"\t\t",item[1]*coeffcients[0]+item[2]*coeffcients[1]+item[3]*coeffcients[2] + intercept)
