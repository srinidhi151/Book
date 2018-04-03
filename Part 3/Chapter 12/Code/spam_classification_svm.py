import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df=pd.read_csv("spam.csv",encoding='latin-1') #Read in the data into a dataframe
df=df[['v1','v2']]
df.columns=['label','sms']

#Function to split the text messages into into words
def create_lexicon(sent):
    sent=re.sub("[^a-zA-Z]"," ",sent)
    sent=sent.lower()
    all_words = word_tokenize(sent)
    lex=list(all_words)
    lex = [lemmatizer.lemmatize(i) for i in lex]   #Converting each word to its root word
    lex = [w for w in lex if not w in stop_words]  #Removing stop words
    lex = [w for w in lex if(len(w)>1)]            #Removing single letter words
    return (" ".join(lex))

rows,col=df.shape

split_msg=[]                                       #Creating new column in the dataframe
for i in range(0,rows):                            #which contains the lemmatized words of
    broken=create_lexicon(df['sms'][i])          #each message with the stop words removed
    split_msg.append(broken)
df['split_msg']=split_msg

#Splitting data into training and testing set, 80% training data and 20% testing data
X_train,X_test,y_train,y_test=train_test_split(df['split_msg'],df['label'],test_size=0.2)


train_data=vectorizer.fit_transform(X_train)
train_data=train_data.toarray()

test_data=vectorizer.transform(X_test)
test_data=test_data.toarray()

model=svm.SVC(kernel='linear')
model.fit(train_data,y_train)

#Testing the model

predicted = model.predict(test_data)
print ("Accuracy")
print (accuracy_score(y_test, predicted))