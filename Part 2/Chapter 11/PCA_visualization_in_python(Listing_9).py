#import statements
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Visualize 3d data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z=features_3d[:,0:1],features_3d[:,1:2],features_3d[:,2:]
ax.scatter(xs=x,ys=y,zs=z,marker='^')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()

#train the PCA model
model=PCA(n_components=2)
model.fit(features_3d)
features_2d=model.transform(features_3d)

#Visualize 2d data
x,y=features_2d[:,0:1],features_2d[:,1:2]
plt.scatter(x,y,color='b',marker='^')
plt.show()
