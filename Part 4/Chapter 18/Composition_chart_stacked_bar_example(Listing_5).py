import numpy as np
import matplotlib.pyplot as plt

N = 5 #Number of groups
#Random scores for X and Y
X = (20, 35, 30, 35, 27)
Y = (25, 32, 34, 20, 25)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bar

p1 = plt.bar(ind, X, width)
p2 = plt.bar(ind, Y, width,bottom=X)

plt.ylabel('Scores')
plt.title('groups')
plt.xticks(ind, ('1', '2', '3', '4', '5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('X', 'Y'))

plt.show()
