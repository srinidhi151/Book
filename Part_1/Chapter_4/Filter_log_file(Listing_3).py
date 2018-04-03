from pyspark.context import SparkContext
from pyspark.sql import SQLContext,Row
sc=SparkContext()
spark=SQLContext(sc)
#load the data(the log file)
data=sc.textFile("log_file.txt")
#create a single column called line where each row re-fers to the lines in the log file
df = data.map(lambda r: Row(r)).toDF(["line"])
#filter out the lines which do not have the word error in them
df = df.filter("line like '%error%'")  # line is the column name and %error% matches the substring error just like an sql query
print("Number of errors = ",df.count())
print("The log file related to the errors are \n",df.collect())
