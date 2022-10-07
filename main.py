import numpy as np
import pandas as pd
from Testing import KNN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics


def TF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def IDF(documents, len):
    import math
    N = len
    # print(N)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        # print(float(val))
        idfDict[word] = math.log(N / float(val))
        # print(idfDict[word])

    return idfDict


def TFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, tf in tfBagOfWords.items():
        tfidf[word] = tf * idfs[word]
    return tfidf


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy



A = "The rain is pouring the"
B = "The rain has stopped"
C = "The sun is up The sun is set The sun is round Rain and sun are the best"
Dataset = {
    'data': ['Why sun is up at east and set at west?',
             'There is only one sun and one moon at earth',
             'i love sunny day',
             'why people always skeptical when see sun is up at west, thats what i thought',
             'Man this life is suck'],
    'intent': [1, 2, 2, 1, 2]
    # 1 is Asking
    # 2 is Statement
}
A = A.lower()

bowDataset = {'data': []}

for sentence in Dataset['data']:
    bowDataset['data'].append(sentence.lower().split(' '))

uniqueWords = set()
for word in bowDataset['data']:
    uniqueWords = uniqueWords.union(set(word))


numOfWordsDataset = dict.fromkeys(uniqueWords, 0)

for sentence in bowDataset['data']:
    for word in sentence:
        numOfWordsDataset[word] += 1

tfDataset = {'data': []}
for sentence in bowDataset['data']:
    numOfSentenceDataset = dict.fromkeys(sentence,0)
    for word in sentence:
        numOfSentenceDataset[word] += 1
    tfTemp = TF(numOfSentenceDataset,sentence)
    tfDataset['data'].append(tfTemp)
idfs = {'data': []}
tfidf = {'data': []}
for sentence in tfDataset['data']:
    idfsTemp = IDF([numOfWordsDataset,sentence],len(Dataset['data']))
    idfs['data'].append(idfsTemp)

    tfidfTemp = TFIDF(sentence,idfsTemp)
    tfidf['data'].append(tfidfTemp)

value = {'data': []}
for sentence in tfidf['data']:
    value['data'].append(sum(sentence.values()))


X, y = value['data'], Dataset['intent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("KNN classification accuracy", accuracy(y_test, predictions))

