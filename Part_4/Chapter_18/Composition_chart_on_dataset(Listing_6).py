import matplotlib.pyplot as plt
# Data
labels = 'Mum-bai','Bhopal','Delhi','Noida','Hyderabad','Pune','Kolkatta','Lucknow','Chennai','Gangtok','Pondicherry','Bangalore'
sizes = [90,12,15,50,10,22,30,70,80,44,24,78]
fig1, ax = plt.subplots()
ax.pie(sizes, labels=labels,autopct='%1.0f%%', pctdistance=1.1,shadow=True, startan-gle=90,labeldistance=1.3)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')
# Set the title for the chart
ax.set_title('Crime Rate Analysis',y=1.08)
plt.show()
