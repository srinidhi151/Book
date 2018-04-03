from pyspark.context import SparkContext
from pyspark.sql import SQLContext

sc=SparkContext()
spark=SQLContext(sc)
#load the dataset
data=sc.textFile("sampled_purchases.txt")

#choosing jut the columns with the product names and its cost
reviews = spark.createDataFrame(data.map(lambda x: tu-ple([x.split(",")[4],float(x.split(",")[5])])),["Product","cost"])

#use of group by to calculate the average cost of a product
prod-uct_averages=reviews.groupBy("Product").avg().collect()
print(product_averages)
