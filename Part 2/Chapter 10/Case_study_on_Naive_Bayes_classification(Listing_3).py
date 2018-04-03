from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Read the data
data=pd.read_csv("NBOrDT.csv")
print(data)

#Neglect last row as output label is not defined Select columns from income to enroll
dataset=np.array(data.loc[:len(data)-2,'Income':'Enrolls'])
print(dataset)

#Since my dataset contains strings (categorical data),  encode the strings to integers to proceed.
# For this use LabelEncoder

le=LabelEncoder()
for i in range(4):
    le.fit(dataset[:,i])
    dataset[:,i]=le.transform(dataset[:,i])

#Extarct the features and labels now
X=dataset[:,:3]
Y=dataset[:,3]
dataset=list(zip(X,Y))

#Split the dataset into train and test
train , test = train_test_split(dataset, test_size=0.25)
print('The training set consists of '+ str(len(train))+' points')
print('The test set consists of '+ str(len(test))+' points')

#To build the model select the features and labels
train_features = []
train_labels = []
test_features = []
test_labels = []
for item in train:
    train_features.append(item[0])
    train_labels.append(item[1])
for item in test:
    test_features.append(item[0])
    test_labels.append(item[1])

# Naive Bayes model
classifier = GaussianNB()
classifier.fit(train_features, train_labels)
results = classifier.predict(test_features)

#Accuracy of the model
print("model accuracy",accuracy_score(test_labels, results))
print(confusion_matrix(test_labels, results))
