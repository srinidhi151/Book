import sys

cur_sales = 0
cur_product = None

for line in sys.stdin:
    data = line.strip().lower().split('\t')
    if len(data) != 2:
        continue

    product, revenue = data

    if(cur_product != None and cur_product != product):
        print('{0}\t{1}'.format(cur_product,cur_sales))

        cur_sales = 0

    cur_product = product
    cur_sales = cur_sales + float(revenue)

if( cur_product != None):
    print("{0}\t{1}".format(cur_product,cur_sales))

