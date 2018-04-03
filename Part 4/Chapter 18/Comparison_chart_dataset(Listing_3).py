import matplotlib.pyplot as my_plt
import csv
x=[]
y=[]
y1=[]
y2=[]
# Read the data from CSV
with open('mobile_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
        x.append((row[0]))
        y.append((row[1]))
        y1.append((row[2]))
        y2.append((row[3]))

my_plt.plot(x,y, label='Samsung: Mobile Users')
my_plt.plot(x,y1, label='Nokia: Mobile Users')
my_plt.plot(x,y2, label='LG: Mobile Users')
my_plt.locator_params(tight=True,nbins= 55)
my_plt.title('Comparison Graph')
my_plt.xticks(x, rotation='vertical')
# Pad margins
my_plt.margins(0)
# Adjust spacing to prevent clipping of tick labels
my_plt.subplots_adjust(bottom=0.15)
my_plt.legend()
my_plt.show()

