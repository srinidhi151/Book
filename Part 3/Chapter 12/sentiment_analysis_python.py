import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import tensorflow as tf
import pandas as pd
import re
import csv
from wordcloud import WordCloud,STOPWORDS
import matplotlib as mpl
import matplotlib.pyplot as plt


df=pd.read_table('yelp_labelled.txt',names=('review','sentiment'))
df2=pd.read_table('imdb_labelled.txt',names=('review','sentiment'))
df3=pd.read_table('amazon_cells_labelled.txt',names=('review','sentiment'))
df=pd.concat([df,df2])
df=pd.concat([df,df3])

def create_lexicon(sent,lex):
    sent=re.sub("[^a-zA-Z]"," ",sent)
    sent=sent.lower()
    all_words = word_tokenize(sent)
    lex+= list(all_words)
    return list(all_words)


lexicon = []
pos_words=[]
neg_words=[]
for index, row in df.iterrows():
    if(row['sentiment']==1):
        pos_words+=create_lexicon(row['review'],lexicon)
    else:
        neg_words+=create_lexicon(row['review'],lexicon)


lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

w_counts = Counter(lexicon)
l2 = []
for w in w_counts:
    if 2000 > w_counts[w] > 50:
            l2.append(w)
l3=[]
for i in l2:
    if(len(i)>1):
        l3.insert(0,i)


def create_feature(df2,lexicon):
    featureset = []
    for l in df2['review']:
        current_words = word_tokenize(l.lower().decode('utf-8'))
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        for word in current_words:
            if word.lower() in lexicon:
                index_value =lexicon.index(word.lower())
                features[index_value] += 1

        features = list(features)
        featureset.append(features)

    return featureset
X_train=create_feature(df[:2500],l3)
X_test=create_feature(df[2500:],l3)


y_train=list(df['sentiment'][:2500])
for i in range(len(y_train)):
        l=[0]*2
        l[int(y_train[i])]=1
        y_train[i]=l
y_test=list(df['sentiment'][2500:])
for i in range(len(y_test)):
        l=[0]*2
        l[int(y_test[i])]=1
        y_test[i]=l


wordcloud = WordCloud(background_color='white',
                          stopwords=STOPWORDS,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(pos_words))
print wordcloud
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Positive word cloud")
plt.show()
 
 
wordcloud = WordCloud(background_color='white',
                          stopwords=STOPWORDS,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(neg_words))
print wordcloud
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Negative word cloud")
plt.show()

n_nodes_hl1 = 2000
n_nodes_hl2 = 2000
n_nodes_hl3 = 2000

n_classes = 2
batch_size = 100
hm_epochs = 500

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(X_train[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


def nnmodel(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train(x):
    pred = nnmodel(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(X_train):
                start = i
                end = i+batch_size
                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += c
                i+=batch_size
                
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:X_test, y:y_test}))
        return pred


p=train(x)

