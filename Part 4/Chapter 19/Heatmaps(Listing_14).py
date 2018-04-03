import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("train.csv")

#Heatmap of Indoor Characteristics
df_indoor=df[['price_doc','full_sq','life_sq','floor','max_floor','state','kitch_sq','num_room']]
corrmat = df_indoor.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,cbar=True, annot=True, square=True);
df_indoor.fillna(method='bfill',inplace=True)
plt.show()
