#import statements
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load the cars dataset.
data = pd.read_csv('mtcars.csv')
attributes = ['mpg', 'disp', 'hp', 'wt']
data = data[attributes]
train, test = train_test_split(data, test_size=0.10)

# Transform the train and test data into features and targets.
train_X = np.array(train[['disp', 'hp', 'wt']])
train_Y = np.array(train[['mpg']])
test_X = np.array(test[['disp', 'hp', 'wt']])
test_Y = np.array(test['mpg'])

# Train and test the Ridge regression model.
model = Ridge()
model.fit(train_X, train_Y)
results = model.predict(test_X)

# Metrics.
print(" Mean Squared error : {0}".format(mean_squared_error(test_Y, results)))
print("Actual Value\tPredicted value")
for i in range(len(results)):
    print("{0}\t\t\t{1}".format(test_Y[i], results[i]))
