#import statements
from apyori import apriori

#get the transactions
transactions = [
    ['A','B','C'],
    ['A','C'],
    ['B','C'],
    ['A','D'],
    ['A','C','D']

]

#apply apriori
results = list(apriori(transactions,min_support = 0.5))
print(results[0])
print(results[1])
print(results[2])
