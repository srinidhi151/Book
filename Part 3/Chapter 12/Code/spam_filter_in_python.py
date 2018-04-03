from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

dataset=pd.read_csv("spam.csv",encoding='latin')
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1)
dataset.head(10)

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


tfidf_vectorizer = TfidfVectorizer("english")
tfidf_vectorizer.fit(content)
features = tfidf_vectorizer.transform(content)
features = features.todense()
features_train, features_test, labels_train, labels_test = train_test_split(features, dataset['v1'], test_size=0.3,shuffle=True)

model=GaussianNB()
model.fit(features_train,labels_train)
GaussianNB(priors=None)
 
test=dataset.sample(10).copy()
test_features=test['v2']
test_lables=test['v1']
test.rename(columns={'v1':'Actual_Class','v2':'Email Content'},inplace=True)
test=test.reindex_axis(['Email Content','Actual_Class'],axis=1)
print test

test_features=test_features.apply(preprocess)
test_features=tfidf_vectorizer.transform(test_features)
test_features=test_features.todense()
model.predict(test_features)
test['Predicted_Class']=model.predict(test_features)
print test
 
print("The confusion matrix:-\n",confusion_matrix(labels_test,model.predict(features_test)))
print("accuracy ",accuracy_score(labels_test,model.predict(features_test)))

