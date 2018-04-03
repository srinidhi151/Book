import matplotlib.pyplot as plt
import numpy as np

#Random Value generation
x=np.random.rand(50)
y=np.random.rand(50)
c=np.random.rand(50)
#Bubble size dependent on c, more the data larger the bubble
plt.scatter(x,y,s=500*c,label="point") 
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Bubble Plot")
plt.legend()
plt.show()
