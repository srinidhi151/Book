import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("train.csv")
df_indoor=df[['price_doc','full_sq','life_sq','floor','max_floor','state','kitch_sq','num_room']]
df_indoor=df[df.max_floor<20]
plt.figure(figsize=(16,8))
sns.boxplot(x="max_floor", y="price_doc", data=df_indoor)
plt.show()
