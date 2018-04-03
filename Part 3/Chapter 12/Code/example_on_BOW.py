from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
 
#load dataset
Text = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
]
 
def preprocess(sentence):
    sentence=sentence.lower()
    sentence="".join([x for x in sentence if x not in string.punctuation])
    sentence=[x for x in sentence.split(" ") if x not in stopwords.words('english')]
    sentence=[x for x in sentence if x!='']
    return " ".join(sentence)
 
bog_vectorizer = CountVectorizer(lowercase=True,preprocessor=preprocess)
model = bog_vectorizer.fit(Text)
 
bag_of_words=model.transform(Text)
print(bag_of_words.todense())
print(model.get_feature_names())