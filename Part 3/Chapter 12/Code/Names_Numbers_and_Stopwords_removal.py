import string
from nltk.corpus import stopwords
import re

sentence="PretzelBros, airbnb for people who like pretzels, raises $2 million"

sentence=sentence.lower()
sentence

#'pretzelbros, airbnb for people who like pretzels, raises $2 million'

symbols=string.punctuation
sentence="".join([x for x in sentence if x not in symbols])
sentence

#'pretzelbros airbnb for people who like pretzels raises 2 million'

sentence=" ".join([x for x in sentence.split() if x not in stopwords.words('english')])

sentence=re.sub('[0-9]',"",sentence)
print(sentence)
#pretzelbros airbnb people like pretzels raises  million
