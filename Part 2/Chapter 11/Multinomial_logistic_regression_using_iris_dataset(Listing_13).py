#import the required libraries for model building and data handling.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

#Load the dataset and generate the train and test datasets with features and labels.
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

#Build the model with Newton conjugate gradient solver for multinomial classification.
model = LogisticRegression(solver='newton-cg') # since, multiclass
model.fit(train_features,train_labels)

#Get the model parameters.
print(model.coef_,model.intercept_)

#test the model.
results = model.predict(test_features)
acc = accuracy_score(test_labels,results)
cm = confusion_matrix(test_labels,results,labels=[0,1,2])
print(acc)

#print the confusion matrix.
for row in cm:
    for item in row:
        print(item, end='\t')
    print()
