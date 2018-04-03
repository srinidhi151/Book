
from pyspark.context import SparkContext
import random
sc=SparkContext()

def func(p):
    x, y = random.random(), random.random()
    if x**2 + y**2 < 1:
        return 1
    else :
        return 0
samples=20000
df = sc.parallelize(range(0, samples)).filter(func)
count=df.count()
print("Pi is around ",4.0 * count / samples)
