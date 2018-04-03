#import statements
from matplotlib import pyplot as plt
from matplotlib import image 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

#Visualize the trees
trees=[]
for tree in classifier.estimators_:
    dot = StringIO()
    export_graphviz(tree, out_file=dot,filled=True, rounded=True,feature_names=data.feature_names,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot.getvalue())  
    trees.append(graph)

tree_number=1
Image(trees[tree_number].create_png())
