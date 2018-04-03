#import statements
from pandas import read_csv
import numpy

#Load the data
data = read_csv('pima-indians-diabetes.data.csv', header=None)
print(data)

#0 is unaccpetable value so it is considered as missing.We replace 0 with Nan  
data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, numpy.NaN)

# then we count the number of NaN values in each column to decide the strategy to handle to missing value
print(data.isnull().sum())

#we can handle the missing data by removing the rows with missing data
data.dropna(inplace=True)
