from pyspark.context import SparkContext
from pyspark.sql import SQLContext

sc=SparkContext()
spark=SQLContext(sc)
#load the dataset
data=sc.textFile("text_file.txt")
data.cache()  #caches the read text file for better lookup speeds
#counting the number of occurances of each characters
count=data.flatMap(lambda x : list(x)).map(lambda char-acter: (character, 1)).reduceByKey(lambda a, b: a + b)
print(count.collect())
