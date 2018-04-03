
# coding: utf-8

# In[4]:


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


# In[55]:


f=open('train_questions.txt','rU')   #Reads in the training questions text file
with open('train_labels.csv','rb') as k:
    reader=csv.reader(k)
    train_labels=list(reader)                                         #Reads in the training labels file


# In[56]:


df=pd.read_table('train_questions.txt',sep=',',names=('ID','Ques'))
df=df.iloc[1:]


# In[57]:


df_label=pd.DataFrame(train_labels)
df_label=df_label.iloc[1:]
df_label.columns=['ID','Label']


# In[61]:


df=pd.merge(df,df_label,on='ID')


# In[73]:


def create_lexicon(sent,lex):
    sent=re.sub("[^a-zA-Z]"," ",sent)
    sent=sent.lower()
    all_words = word_tokenize(sent)
    lex+= list(all_words)


lexicon = []
for i in df.Ques:
    create_lexicon(i,lexicon)


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
    for l in df2['Ques']:
        current_words = word_tokenize(l.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        features = list(features)
        featureset.append(features)

    return featureset


# In[75]:


X_train=create_feature(df[:2500],l3)
X_test=create_feature(df[2500:],l3)


# In[76]:


y_train=list(df['Label'][:2500])
for i in range(len(y_train)):
        l=[0]*6
        l[int(y_train[i])]=1
        y_train[i]=l
y_test=list(df['Label'][2500:])
for i in range(len(y_test)):
        l=[0]*6
        l[int(y_test[i])]=1
        y_test[i]=l


# In[80]:


print len(X_train)
print len(y_train)
print len(X_test)
print len(y_test)


# In[85]:


hiddden_layer_1 =2500
hidden_layer_2 = 2500
hidden_layer_3 = 2500

n_classes = 2
batch_size = 100
epochs = 500

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f':hiddden_layer_1,
                  'weight':tf.Variable(tf.random_normal([len(X_train[0]), hiddden_layer_1])),
                  'bias':tf.Variable(tf.random_normal([hiddden_layer_1]))}

hidden_2_layer = {'f':hidden_layer_2,
                  'weight':tf.Variable(tf.random_normal([hiddden_layer_1, hidden_layer_2])),
                  'bias':tf.Variable(tf.random_normal([hidden_layer_2]))}

hidden_3_layer = {'f':hidden_layer_3,
                  'weight':tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_3])),
                  'bias':tf.Variable(tf.random_normal([hidden_layer_3]))}

output_layer = {'f':None,
                'weight':tf.Variable(tf.random_normal([hidden_layer_3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


def layers(data):

    layer_1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3,output_layer['weight']) + output_layer['bias']

    return output

def train_model(x):
    pred = layers(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            i=0
            while i < len(X_train):
                start = i
                end = i+batch_size
                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, k = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += k
                i+=batch_size
                
            print('Epoch', epoch+1, 'completed out of',epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:X_test, y:y_test}))
        return pred


# In[86]:


p=train_model(x)


# In[86]:


p=train(x)


# In[87]:


X_train


# In[ ]:




