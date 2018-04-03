import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('activity-data.csv')
print(df)

ax = \
df[["RightLeg_X-axis_accelerometer", "RightLeg_Y-axis_accelerometer", "RightLeg_Z-axis_accelerometer"]].plot(title = "(X,Y,Z) Right Leg Acceleration Measurements vs. Time",
                                           figsize=(16,5));

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Acceleration");
plt.show()


ax = \
df[["LeftLeg_X-axis_gyroscope", "LeftLeg_Y-axis_gyroscope", "LeftLeg_Z-axis_gyroscope"]].plot(title = "(X,Y,Z) Left Leg Gyroscope Measurements vs. Time",
                                        figsize=(15,5));
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Acceleration");
plt.show()

 

g = sns.PairGrid(df[["Torso_X-axis_accelerometer", "Torso_Y-axis_accelerometer", "Torso_Z-axis_accelerometer"]],size =2.5, aspect=2.0)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(sns.kdeplot, lw=3, legend=False);
plt.show()
