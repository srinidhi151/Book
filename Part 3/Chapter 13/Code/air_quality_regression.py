import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing,model_selection
from sklearn.linear_model import  LinearRegression

csv_file=open('AirQualityUCI_req.csv','r')
data = list(csv.DictReader(csv_file))

attr_list=['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
matrix=np.zeros([9357,len(attr_list)])
print(data[1]['CO(GT)'])
i=0
j=0
try:
    for item in data:
        for attr in attr_list:

            matrix[i][j]=float(item[attr])


            j=j+1
        i=i+1
        j=0
except Exception:
    pass

dframe=pd.DataFrame(matrix,columns=attr_list)

x=np.array(dframe['T'].values.reshape(9357,1))
y=np.array(dframe['C6H6(GT)'].values.reshape(9357,1))

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.99)

clf=LinearRegression()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print("Accuracy: "+str(accuracy))

plt.scatter(x_train,y_train,color='black')
pred=clf.predict(x_train)
plt.plot(x_train,pred,color='blue')
plt.xlim(0,40)
plt.ylim(0, 40)
plt.show()

y_axes=np.concatenate([np.array(i) for i in y_train])
y_pred=np.concatenate([np.array(i) for i in pred])
X=[]
for i in range(len(y_axes)):
    X.append([y_axes[i],y_pred[i],y_axes[i]-y_pred[i]])
print(X)
