#import the modules required.
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize,word_tokenize

#Load the corpus from a text file and tokenize it into sentences.
with open('matter.txt','r') as f:
    data = f.read()
Text = sent_tokenize(data)
# Total number of sentences in the data. Prints 14 for this text.
print(len(Text)) 

#Define the preprocessor routine for the data.
def preprocess(sentence):
    sentence=sentence.lower()
    sentence="".join([x for x in sentence if x not in string.punctuation])
    sentence=[x for x in sentence.split(" ") 
if x not in stopwords.words('english')]
    sentence=[x for x in sentence if x!='']
    return " ".join(sentence)

# Fit a bag of words estimator and transform the count matrix.
bow_vectorizer = CountVectorizer(lowercase=True,preprocessor=preprocess)
model = bow_vectorizer.fit(Text)
bag_of_words=model.transform(Text)

#Get the frequencies of the words.
bow = bag_of_words.todense()
#Get the words in the corpus.
words = model.get_feature_names()

#See the details of the estimator values.
print(bow.shape)   # prints (14, 159)
print(len(words))  # prints 159
