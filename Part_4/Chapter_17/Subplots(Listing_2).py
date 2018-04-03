import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
x=[1,3,2,4,2]
y=[1,2,3,4,5]
#block 1
axs[0, 0].plot(x,y,marker="*")
x1 = [2,3,4,5]
y1 = [1,6,3,2]
#block 2
axs[1, 0].scatter(x1,y1)
x_bar=["Python","Java","c++","c","R","Julia"]
y_bar=[10,7,5,1,8,9]
ypos=[i for i in range(1,len(x_bar)+1)]
axs[0, 1].bar(ypos,y_bar,align='center') #block 3
axs[1, 1].pie([1,2,3],explode=[0.2,0.1,0.15]) #block 4
plt.show()
