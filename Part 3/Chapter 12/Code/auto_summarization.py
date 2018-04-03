from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
puncs =  (list(punctuation))

with open('matter.txt','r') as f:
    content = f.read()
sentences = sent_tokenize(text=content.lower())

words = []
for i in sentences:
    words.extend(word_tokenize(i))

stop = set(stopwords.words('english') + puncs )

final_words = []
for word  in words:
    if word not in stop:
        final_words.append(word)

table = FreqDist(final_words)
ranked_words = sorted(table,key=table.get)



sent_ranks = {}
for sent in sentences:
    w = word_tokenize(sent)
    rank = 0
    for word in w:
        if word in ranked_words:
            rank = rank + ranked_words.index(word)
    sent_ranks[rank] = sent 


final_sents = sorted(sent_ranks.items())
final_sents.reverse()
final = []
for item in final_sents[0:10]:
    final.append(sentences.index(item[1]))

for index in sorted(final):
    print(sentences[index] + '\n')
