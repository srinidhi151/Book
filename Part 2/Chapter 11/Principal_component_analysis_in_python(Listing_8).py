#import statements
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
import numpy as np

#Load the data
data = load_diabetes()
#dataset = list(zip(data.data, data.target))
features=np.array(data.data)
print("Number of dimensions of the dataset is ",features.shape[1])

#Fit the PCA model
model=PCA(n_components=2)
model.fit(features)

print("10 dimensional features is converted to a two dimensional features\nFor instance",features[0:1,:],"->",model.transform(features[0:1,:]))

model.explained_variance_ratio_

feature_in_2d=model.transform(features[0:1,:])
feature_in_10d=model.inverse_transform(feature_in_2d)
print("Two dimensional values =",feature_in_2d)
print("Ten dimensional values = ",feature_in_10d)
print("Original Ten dimensional values = ",features[0:1,:])

features_3d=features[:,0:3]
features_3d.shape[1]
