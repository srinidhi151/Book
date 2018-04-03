#import statments
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Load the dataset
data = load_iris()
dataset = list(zip(data.data, data.target))

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
