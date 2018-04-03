#Histogram with a best fit line

import numpy as np
import matplotlib.mlab as mlab #For the best fit line
import matplotlib.pyplot as plt
 
 
# Random example data
mu = 100 # mean of distribution
sigma = 15 # standard deviation of distribution
x = mu + sigma * np.random.randn(100000) #Normalised data
 
num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
 
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')

#Histogram details
plt.xlabel('Data Point')
plt.ylabel('Probability')
plt.title(r'Sample Histogram')
 
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
