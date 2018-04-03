import nltk
import csv
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC

f=open('train_questions.txt','rU')   
with open('train_labels.csv','rb') as k:
    reader=csv.reader(k)
    train_labels=list(reader)                                         #Reads in the training labels file
train_labels.remove(train_labels[0])                                  #removes 'id' and 'label' from the label file

train_data=f.read()

train_sent=train_data.splitlines()  
train_sent.remove(train_sent[0])                                          #split the training set into its corresponding 
#print len(train_sent)                                        
final_set=[]
all_words1=[]
token=nltk.RegexpTokenizer(r'\w+')                  #the word tokenizer that does not read in punctuation
all_words=token.tokenize(train_data)                #All words in the file are tokenized
for j in all_words:                                  
    if j.isdigit() is False:                        #Read in only non numerical words present in the entire train set
        all_words1.append(j)
e=0
for i in train_sent:                    # Creates a list of list of lists with words of each question and the 
    words=[]                            # corresponding label [0-6]
    set1=[]
    set2=[]
    words=nltk.word_tokenize(i)
    set1.append(words[2:]) 
    set1.append(train_labels[e][1])
    final_set.append(set1)
    e=e+1

all_words2=nltk.FreqDist(all_words1)    #The frequency distribution of all of the words present in the train file
word_features=list(all_words2.keys())
#print len(word_features)

def find_features(sent):                # Finding the features of each question and storing it as a dictionary
    words2=set(sent)
    features={}
    for w in word_features:
        features[w]=(w in words2)
    return features

featuresets=[(find_features(rev),category) for (rev, category) in final_set]
# Finds all the features of all the questions present in the training set and puts it in the form of a list

training_set=featuresets[:2900]
testing_set=featuresets[2900:]
#Split of 80:20 for training and testing set

print "Training Classifier ......"
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print "Accuracy"
print nltk.classify.accuracy(LinearSVC_classifier, testing_set)