#import statements
import numpy as np
import collections
import pydotplus
from sklearn import tree
import graphviz

# Data Collection
X = [ [0,0],
      [1,0],
      [1,1],
      [2,1],
      [2,1],
      [2,0]]

rain = np.array(['not rain', 'not rain', 'not rain', 'rain', 'rain', 'not rain'])
data_feature_names = [ 'weather_type', 'atmospheric_pressure']

# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,rain)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
