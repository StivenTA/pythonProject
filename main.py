import numpy as np
import pandas as pd
from Testing import KNN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics
import string

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



A = 'The rain is pouring the'
B = 'The rain has stopped'
C = 'The sun is up The sun is set The sun is round Rain and sun are the best'
Dataset = {
    'data': ['Saya siapa?',
             'siapa dosen yang akan mengajar pada semester ini?',
             'i love sunny day',
             'why people always skeptical when see sun is up at west, thats what i thought',
             'Man this life is suck',
             'The sun is up The sun is set The sun is round Rain and sun are the best',
             'The rain has stopped',
             'The rain is pouring the'],
    'intent': [1, 2, 2, 1, 2, 2, 2, 2]
    # 1 is Asking
    # 2 is Statement
}
A = A.lower()

bowDataset = {'data': []}

for sentence in Dataset['data']:
    # .translate(str.maketrans('','',string.punctuation)) digunakan untuk menghilangkan tanda baca
    bowDataset['data'].append(sentence.translate(str.maketrans('','',string.punctuation)).lower().split(' '))
print(bowDataset)
uniqueWords = set()
for word in bowDataset['data']:
    #untuk mencari semua kata pada dataset
    uniqueWords = uniqueWords.union(set(word))


numOfWordsDataset = dict.fromkeys(uniqueWords, 0)

for sentence in bowDataset['data']:
    for word in sentence:
        #mencari jumlah setiap kata muncul dari seluruh dataset
        numOfWordsDataset[word] += 1

tfDataset = {'data': []}
for sentence in bowDataset['data']:
    numOfSentenceDataset = dict.fromkeys(sentence,0)
    for word in sentence:
        #mencari jumlah kata muncul dari sebuah kalimat
        numOfSentenceDataset[word] += 1
    #Menghitung nilai Term Frequency untuk setiap kalimat
    tfTemp = TF(numOfSentenceDataset,sentence)
    tfDataset['data'].append(tfTemp)
    # print(numOfSentenceDataset)
idfs = {'data': []}
tfidf = {'data': []}

for sentence in tfDataset['data']:
    # mencari nilai IDF pada setiap kalimat dataset
    # len diambil untuk menghitung total dari kalimat yang ada
    # IDF menggunakan numOfWordsDataset karena menghitung setiap kata pada kalimat pada seluruh kata unik yang ada
    idfsTemp = IDF([numOfWordsDataset,sentence],len(Dataset['data']))
    tfidfTemp = TFIDF(sentence,idfsTemp)
    tfidf['data'].append(tfidfTemp)

value = {'data': []}
for sentence in tfidf['data']:
    # nilai tfidf diubah menjadi nilai rata-rata
    value['data'].append(statistics.mean(sentence.values()))
print(value)


X, y = value['data'], Dataset['intent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("KNN classification accuracy", accuracy(y_test, predictions))