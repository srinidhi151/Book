from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
puncs =  (list(punctuation))
puncs.extend(["'s","''","``"])

with open('files/matter.txt') as f:
    data = f.read()
wordcloud = WordCloud()
wordcloud.generate(data)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


with open('files/matter.txt') as f:
    data = f.read()
sentences = sent_tokenize(text=data.lower().strip())


words = []
for i in sentences:
    words.extend(word_tokenize(i))


stop = set(stopwords.words('english') + puncs )
final_words = []
for word  in words:
    if word not in stop:
        final_words.append(word)
table = FreqDist(final_words)

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(table)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
