#import statements
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import graphviz

#Prepare the data
data = load_iris()
dataset = list(zip(data.data, data.target))
train , test = train_test_split(dataset, test_size=0.25)
print(len(train))
print(len(test))
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
print(train_features, train_labels, test_features, test_labels )

#train the model
classifier = DecisionTreeClassifier()
classifier.fit(train_features, train_labels)
results = classifier.predict(test_features)
print(accuracy_score(test_labels, results))

#Visualize the tree
with open('iris_dt.dot', 'w') as f:
	export_graphviz(classifier, out_file=f,  
                          feature_names=data.feature_names,  
                           class_names=list(data.target_names))

with open('iris_dt.dot', 'r') as f:
	dot_graph = f.read()
	graphviz.Source(dot_graph).view()
