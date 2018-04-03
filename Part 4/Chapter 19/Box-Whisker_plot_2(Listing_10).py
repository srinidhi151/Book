#Single box plots
import matplotlib.pyplot as plt
import numpy as np

# Random data #
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
#             #

plt.boxplot(data) # Vertical box plot
plt.figure()
plt.boxplot(data, 0, 'rs', 0) #Horizontal box plot
plt.show()
