from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
def mse(a,b):
    sum = 0
    for i in range(len(a)):
        sum = sum + (a[i]-b[i])**2
    return sum/len(a)
sc = SparkContext()

#load the data
data = sc.textFile('als_data.csv').map(lambda x: x.split(','))
print(data.collect())

#get the rankings of each user input.
movie_input = data.map(lambda x : Rating(int(x[0]), int(x[1]), int(x[2])))

#Train the Alternate Least Squares Model.
latent_facctors = 10
iterations = 20
movie_recommender = ALS.train(movie_input, latent_facctors, iterations)

# Predict the model over the training dataset itself.
test_ds = data.map( lambda x : (x[0], x[1]))
result = movie_recommender.predictAll(test_ds)

# convert the input  predictions into a tuple of length 3.
a = movie_input.map(lambda x : (x[0],x[1],x[2])).collect()
r = result.map(lambda x : x[2] ).collect()


# get the MSE loss
A = movie_input.map(lambda x : x[2]).collect()
r_forLoss = result.map(lambda x : x[2]).collect()
MSE = mse(A,r_forLoss)
print("MSE loss : {0}".format(MSE))

# Print the results.
print("User \t Movie \t actual rating \t predicted rating")
for i in range(len(r)):
    print("{0}\t{1}\t{2}\t\t\t{3}".format(a[i][0],a[i][1],a[i][2],round(r[i])))
