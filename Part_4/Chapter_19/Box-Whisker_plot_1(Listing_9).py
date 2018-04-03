#Multiple box plots
import matplotlib.pyplot as plt
import numpy as np
# Random data 
data = np.random.rand(1000)
spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
data.shape = (-1, 1)
d2.shape = (-1, 1)
data = [d2, d2[::2, 0]]

#Plot
plt.figure()
plt.boxplot(data)
plt.show()
