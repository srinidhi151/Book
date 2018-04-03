import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing,model_selection
from sklearn.linear_model import  LinearRegression

csv_file=open('AirQualityUCI_req.csv','r')
data = list(csv.DictReader(csv_file))

attr_list=['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
matrix=np.zeros([9357,len(attr_list)])
i=0
j=0
for item in data:
    for attr in attr_list:
        matrix[i][j]=item[attr]
        j=j+1
    i=i+1
    j=0
dframe=pd.DataFrame(matrix,columns=attr_list)
n_attr=len(attr_list)
min_attr=np.zeros([n_attr])
max_attr=np.zeros([n_attr])
attr_values=np.zeros([13,9357])

for i in range(13):
    attr_values[i]=dframe[attr_list[i]]

for i in range(n_attr):
    #print(matrix[i])
    min_attr[i] = np.min(attr_values[i])
    max_attr[i] = np.max(attr_values[i])

print(min_attr)
print(max_attr)

for i in range(len(attr_values)):
    attr_values[i]=(attr_values[i]-min_attr[i])/(max_attr[i]-min_attr[i])
#print(attr_values)
print(attr_values.shape)
attr_values_new=attr_values.transpose()
print(attr_values_new.shape)
#print(attr_values)
df=pd.DataFrame(attr_values_new,columns=attr_list)
axes = scatter_matrix(df, alpha=0.2, figsize=(45, 30),diagonal='histo')
corr = df.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.show()
