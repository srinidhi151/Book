import numpy as np
import matplotlib.pyplot as plt
 
x1 = (0.6 + 0.6 * np.random.rand(100), np.random.rand(100))
x2 = (0.4+0.3 * np.random.rand(100), 0.5*np.random.rand(100))
x3 = (0.3*np.random.rand(100),0.3*np.random.rand(100))
 
data = (x1, x2, x3)
colors = ("red", "green", "blue")
groups = ("X1", "X2", "X3") 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
 
for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecol-ors='none', s=30, label=group)
 
plt.title('Scatter plot with groups')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc="upper left")
plt.show()
