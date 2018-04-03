import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
df=pd.read_csv("train.csv")
df_indoor=df[['price_doc','full_sq','life_sq','floor','max_floor','state','kitch_sq','num_room']]
df_indoor=df[df.max_floor<20]
plt.figure(figsize=(10,8))
x=df_indoor.price_doc
y=df_indoor.full_sq
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contourf(X, Y, Z, 20, color='black')
plt.title('House interior feature contour')
plt.show()
