import matplotlib.pyplot as plt
import numpy as np 

#Number of groups of data and arbitary data to be plotted
number_groups=4
means_ex1=(100,20,42,23)
means_ex2=(54,63,24,52)

#Graph parameters
fig,ax=plt.subplots()
index=np.arange(number_groups)
bar_width=0.3
opacity=0.85

#The defination for the two graphs 
rects1=plt.bar(index,means_ex1,bar_width,alpha=opacity,color='b',label='ex1')
rects1=plt.bar(index+bar_width,means_ex2,bar_width,alpha=opacity,color='c',label='ex2')

#X and Y labels and legend definations
plt.xlabel('Case No.')
plt.ylabel('Value')
plt.title('Sample Comparison Chart')
plt.xticks(index+bar_width,('1','2','3','4'))
plt.legend()
plt.tight_layout()
plt.show()
