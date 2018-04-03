#import statements
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# load the dataset
data = load_iris()
dataset = list(zip(data.data[:,:2], data.target))
print(len(dataset))
#print(dataset)
train,test = train_test_split(dataset, test_size=0.30)
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
#print(train_features, train_labels, test_features, test_labels )

#train the model.
model = SVC(kernel='rbf')
model.fit(train_features,train_labels)
results = model.predict(test_features)

#visualization.

train_features = data.data[:,:2]
train_labels = data.target
x_min, x_max = train_features[:, 0].min() - 1 , train_features[:, 0].max() + 1
y_min, y_max = train_features[:, 1].min() - 1, train_features[:, 1].max() + 1
step = (x_max / x_min)/100
mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
plot_features = np.c_[mesh_x.ravel(), mesh_y.ravel()]

Z = model.predict(plot_features)
Z = Z.reshape(mesh_x.shape)

plt.contourf(mesh_x, mesh_y, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(train_features[:, 0], train_features[:, 1], c = train_labels)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(mesh_x.min(), mesh_x.max())
plt.title('SVC with RBF kernel')
