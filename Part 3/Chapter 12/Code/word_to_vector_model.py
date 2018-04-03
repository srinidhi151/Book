from nltk.corpus import abc,stopwords
from string import punctuation
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

sents = abc.sents()
#print(sents[:10])
puncs = list(punctuation)
stop = set(stopwords.words('english') + puncs + ["''" , "``"])
processed_sents = []
for sent in sents:
    temp = []
    for word in sent:
        if word not in stop:
            temp.append(word.lower())
    processed_sents.append(temp)
print(processed_sents[:10])

#Output
#[['pm', 'denies', 'knowledge', 'awb', 'kickbacks', 'the', 'prime', 'minister', 'denied', 'knew', 'awb', 'paying', 'kickbacks', 'iraq', 'despite', 'writing', 'wheat', 'exporter', 'asking', 'kept', 'fully', 'informed', 'iraq', 'wheat', 'sales'], ['letters', 'john', 'howard', 'deputy', 'prime', 'minister', 'mark', 'vaile', 'awb', 'released', 'cole', 'inquiry', 'oil', 'food', 'program'], ['in', 'one', 'letters', 'mr', 'howard', 'asks', 'awb', 'managing', 'director', 'andrew', 'lindberg', 'remain', 'close', 'contact', 'government', 'iraq', 'wheat', 'sales'], ['the', 'opposition', 'gavan', 'o', 'connor', 'says', 'letter', 'sent', '2002', 'time', 'awb', 'paying', 'kickbacks', 'iraq', 'though', 'jordanian', 'trucking', 'company'], ['he', 'says', 'government', 'longer', 'wipe', 'hands', 'illicit', 'payments', 'totalled', '290', 'million'], ['the', 'responsibility', 'must', 'lay', 'may', 'squarely', 'feet', 'coalition', 'ministers', 'trade', 'agriculture', 'prime', 'minister', ',"', 'said'], ['but', 'prime', 'minister#', 'says', 'letters', 'show', 'inquiring', 'future', 'wheat', 'sales', 'iraq', 'prove', 'government', 'knew', 'payments'], ['it', 'would', 'astonishing', '2002', 'prime', 'minister', 'i', 'done', 'anything', 'i', 'possibly', 'could', 'preserve', 'australia', 'valuable', 'wheat', 'market', ',"', 'said'], ['email', 'questions', 'today', 'inquiry', 'awb', 'trading', 'manager', 'peter', 'geary', 'questioned', 'email', 'received', 'may', '2000'], ['it', 'indicated', 'iraqi', 'grains', 'board', 'approached', 'awb', 'provide', 'sales', 'service', '".']]

embeddings = Word2Vec(sentences=processed_sents,size=300,min_count=20,workers=4,sg=0,iter=5,hs=0)
print(embeddings.wv.most_similar('government'))
vocab = list(embeddings.wv.vocab)
X = embeddings[vocab]
tsne_model = TSNE(n_components=2)
X_tsne = tsne_model.fit_transform(X)


data = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
data = data[:100] # use only first 100 words.
print(data)

