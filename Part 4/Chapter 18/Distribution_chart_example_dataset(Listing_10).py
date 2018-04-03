import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as my_plt
#mean value
mean = 100
#standard deviation value
sd = 15
x = mean + sd * np.random.randn(10000)
num_bins = 20
# Histogram
n, bins, patches = my_plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
 # add a 'best fit' line
y = mlab.normpdf(bins, mean, sd)
my_plt.plot(bins, y, 'r--')
my_plt.xlabel('Intelligent persons in a Organization')
my_plt.ylabel('Probability')
my_plt.title('Histogram')
 # Adjusting the spacing
my_plt.subplots_adjust(left=0.15)
my_plt.show()
