from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def preprocess(sentence):
    stemmer=PorterStemmer()
    sentence=sentence.lower()
    sentence="".join([x for x in sentence if x not in string.punctuation])
    sentence=[x for x in sentence.split(" ") if x not in stopwords.words('english')]
    sentence=[x for x in sentence if x!='']
    sentence=[stemmer.stem(x) for x in sentence]
    return " ".join(sentence)

content=dataset['v2'].copy()
content=content.apply(preprocess)

x=dataset.copy()
spam=x[x.v1=="spam"]
spam=spam.v2
spam=spam.apply(preprocess)
spam_text=".".join(spam)
wordcloud_spam = WordCloud().generate(spam_text)
plt.imshow(wordcloud_spam)
plt.axis("off")
print("The spam word cloud is:-")
plt.show()


ham=x[x.v1=="ham"]
ham=ham.v2
ham=ham.apply(preprocess)
ham_text=".".join(ham)
wordcloud_ham = WordCloud().generate(ham_text)
plt.imshow(wordcloud_ham)
plt.axis("off")
print("The not spam word cloud is:-")
plt.show()