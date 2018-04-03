from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud

dataset=pd.read_csv("spam.csv",encoding='latin')
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1)


x=dataset.copy()
spam=x[x.v1=="spam"]
spam=spam.v2
spam_text=".".join(spam)
wordcloud_spam = WordCloud().generate(spam_text)
plt.imshow(wordcloud_spam)
plt.axis("off")
print("The spam word cloud is:-")
plt.show()
 
ham=x[x.v1=="ham"]
ham=ham.v2
ham_text=".".join(ham)
wordcloud_ham = WordCloud().generate(ham_text)
plt.imshow(wordcloud_ham)
plt.axis("off")
print("The not spam word cloud is:-")
plt.show()