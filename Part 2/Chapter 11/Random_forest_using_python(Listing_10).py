#import statements
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

#Load the data
data = load_iris()
dataset = list(zip(data.data, data.target))
train , test = train_test_split(dataset, test_size=0.25)
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

#train the model
number_of_trees=7
classifier = RandomForestClassifier(n_estimators=number_of_trees,bootstrap=True,criterion='entropy') 
classifier = classifier.fit(train_features, train_labels)
results = classifier.predict(test_features)

#Metrics
print("model accuracy",accuracy_score(test_labels, results))
print("Confusion matrix :-\n",confusion_matrix(test_labels,results))
