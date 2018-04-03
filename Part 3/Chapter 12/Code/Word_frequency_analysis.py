
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
import  matplotlib.pyplot as plt

Text = ["Computer is a device that can be instructed to perform specified instructions." ,
        "Computer is used to automate manual labor through unmatching instructions execution."]

def preprocess(sentence):
    sentence=sentence.lower()
    sentence="".join([x for x in sentence if x not in string.punctuation])
    sentence=[x for x in sentence.split(" ") if x not in stopwords.words('english')]
    sentence=[x for x in sentence if x!='']
    return " ".join(sentence)

tf_vectorizer = CountVectorizer(lowercase=True,preprocessor=preprocess)
model = tf_vectorizer.fit(Text)

print(model.vocabulary_)

x = [ i for i in range(len(model.vocabulary_)) ]
y = []
x_t = []
for item in model.vocabulary_.keys():
    x_t.append(item)
    y.append(model.vocabulary_[item])

plt.figure(figsize=(30,30))
plt.bar(x,y)
plt.xticks(x,x_t,rotation='vertical')
plt.show()
