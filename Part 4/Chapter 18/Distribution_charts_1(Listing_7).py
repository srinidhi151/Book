import matplotlib.pyplot as plt
import numpy as np

# example data
mu = 100 # mean of distribution
sigma = 15 # standard deviation of distribution
x = mu + sigma * np.random.randn(100000) #Normalised data
 
num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
 
# Marking the Axes
plt.xlabel('Data Point')
plt.ylabel('Probability')
plt.title(r'Sample Histogram')
 
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
