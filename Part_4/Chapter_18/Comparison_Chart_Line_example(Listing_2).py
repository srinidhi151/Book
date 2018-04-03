import matplotlib.pyplot as plt
import numpy as np 

#Arbitry X and Y values for two different items
x1=(2,4,6,8,12,13,23,32)
x2=(3,5,8,9,16,19,22,34)
y1=(0.1,0.6,0.9,2.4,3.6,4.8,5.9,9.4)
y2=(1,1.3,1.4,1.5,1.8,3,5,7.4)
plt.plot(x1,y1)
plt.plot(x2,y2)

#X and Y labels and legend definations
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Line Comparison Chart')
plt.legend(['Line 1','Line 2'],loc='upper left')
plt.show()
