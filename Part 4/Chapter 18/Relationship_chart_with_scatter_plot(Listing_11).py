import matplotlib.pyplot as plt
import numpy as np

x=np.random.randn(1000)
y=np.random.randn(1000)

plt.scatter(x,y,label="data point",color='blue')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.legend(loc="upper right")
plt.show()
