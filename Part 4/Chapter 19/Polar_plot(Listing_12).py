#Polar plot
import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 5, 0.01)
theta = 2*np.pi*r

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  #Radial Ticks
ax.set_rlabel_position(-22.5) 
ax.grid(True)

ax.set_title("polar plot")
plt.show()
