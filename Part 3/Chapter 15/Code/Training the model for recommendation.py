from extract_movie_rating import parse

R = numpy.array(parse())
R=R[:100, :10]
N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)

nR = numpy.dot(nP, nQ.T)

print("Done")
print(R, file=open("actual_rating.txt", "w"))
print(nR, file=open("predicted_rating.txt", "w"))
