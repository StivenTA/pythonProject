import numpy as np
import xlrd
import pandas as pd
import nltk
import spacy
import re
import statistics
import math
from Testing import KNN
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



def TF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def IDF(documents, len):
    N = len
    # print(documents[0].values())
    idfDict = dict.fromkeys(documents[0].keys(),0)
    # print(idfDict)
    for document in documents:
        # print(document)
        for word, val in document.items():
            if val > 0:
                idfDict[word] = val
    # print(idfDict)
    for word,val in idfDict.items():
        idfDict[word] = math.log(N/float(val),10)
    return idfDict


def TFIDF(tfBagOfWords, idfs):
    tfidf = {}
    # print (tfBagOfWords)
    # print(idfs[0])
    for word, tf in tfBagOfWords.items():
        # print(tf, idfs[0][word])
        tfidf[word] = tf * idfs[0][word]
    return tfidf

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = pd.read_excel("Chat_Intent.xlsx")
message = {'data': data['Message'],
           'intent': data['Intent']}

tokenizing = {'data': [],'intent':[]}
# tokenizing = []
factory = StemmerFactory()
stemmer = factory.create_stemmer()
pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))';

# for word in message:
#     if word == 'data':
#         for text in message[word]:
#             text = stemmer.stem(text)
#             tokenizing[word].append(word_tokenize(text))
#     else:
#         for intent in message[word]:
#             tokenizing[word].append(intent)
# print(tokenizing)
for word in message['data']:
    text = word
    text = stemmer.stem(text)
    tokenizing['data'].append(word_tokenize(text))
for intent in message['intent']:
    tokenizing['intent'].append(intent)
stop = set(stopwords.words('indonesian'))
clean_word = {'data': [],'intent':[]}
for sentence in tokenizing:

    if sentence == 'data':
        for each in tokenizing[sentence]:
            temp_sentence = {'data': [], 'intent': []}
            for word in each:
                if word not in stop:
                    temp_sentence['data'].append(word)
            clean_word['data'].append(temp_sentence['data'])
    else:
        for intent in tokenizing[sentence]:
            clean_word['intent'].append(intent)
# print(clean_word['data'])
# for sentence in tokenizing:
#     temp_sentence = {'data':[]}
#     for word in sentence:
#         if word not in stop and word != '':
#             temp_sentence['data'].append(word)
#     clean_word['data'].append(temp_sentence['data'])
# print(len(clean_word['data']))
#

uniqueWords = set()
for text in clean_word['data']:
    uniqueWords = uniqueWords.union(set(text))
# uniqueWords.remove('')
numOfWordsDataset = dict.fromkeys(uniqueWords, 0)


label = preprocessing.LabelEncoder()
intent = pd.DataFrame(clean_word)
intent['intent'] = label.fit_transform(intent['intent'])
intent['intent'] = intent['intent'].astype('category')
# print(intent['intent'])
# print(testingIntent['intent'].unique())
uniqueIntent = set(tokenizing['intent'])
# print(len(uniqueIntent))
uniqueIntent.union(set(tokenizing['intent']))
numofIntentDataset = dict.fromkeys(uniqueIntent,0)
# print(numofIntentDataset)

for intent in tokenizing['intent']:
    numofIntentDataset[intent] += 1
print(numofIntentDataset)

for sentence in clean_word['data']:
    for word in set(sentence):
        numOfWordsDataset[word] += 1
        # mencari jumlah setiap kata muncul dari seluruh dataset

tfDataset = {'data': []}
for sentence in clean_word['data']:
    numOfSentenceDataset = dict.fromkeys(sentence,0)
    for word in sentence:
        # mencari jumlah kata muncul dari sebuah kalimat
        numOfSentenceDataset[word] += 1
    # Menghitung nilai Term Frequency untuk setiap kalimat
    tfTemp = TF(numOfSentenceDataset,sentence)
    tfDataset['data'].append(tfTemp)
# print(tfDataset)
tfidf = {'data': [],'intent':[]}
idf = {'data': [],'intent':[]}

for sentence in tfDataset['data']:
    # mencari nilai IDF pada setiap kalimat dataset
    # len diambil untuk menghitung total dari kalimat yang ada
    # IDF menggunakan numOfWordsDataset karena menghitung setiap kata pada kalimat pada seluruh kata unik yang ada
    idfsTemp = IDF([numOfWordsDataset],len(clean_word['data']))
    idf['data'].append(idfsTemp)

for sentence in tfDataset['data']:
    tfidfTemp = TFIDF(sentence, idf['data'])
    tfidf['data'].append(tfidfTemp)
for intent in intent['intent']:
    tfidf['intent'].append(intent)
value = {'data': [],'intent':[]}
#
for x,y in tfidf.items():
    if x == 'data':
        for word in tfidf[x]:
            if word.values():
                value['data'].append(statistics.mean(word.values()))
            else:
                value['data'].append(0)
    elif x == 'intent':
        for intent in tfidf[x]:
            value['intent'].append(intent)
exportData = pd.DataFrame(value)
# exportData.to_excel("distance_dataset1.xlsx",index=False)
# point_dataset = pd.read_excel("distance_dataset.xlsx")
# point = {'data':[],
#          'intent': []}
# for data in point_dataset['data']:
#     point['data'].append(data)
# for intent in point_dataset['intent']:
#     point['intent'].append(intent)
# print(point)
# for sentence in tfidf['data']:
#     # print(sentence[0])
#     # nilai tfidf diubah menjadi nilai rata-rata
#     if sentence.values():
#         value['data'].append(statistics.mean(sentence.values()))
#     # else:
#     #     value['data'].append(0)
# for intent in intent['intent']:
#     value['intent'].append(intent)
#

# X, y = point['data'], point['intent']
# for i in range(3,15,2):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=1000
#     )
#     k = i
#     clf = KNN(k=k)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     print("KNN classification accuracy in (k = " + str(i) + "): ", accuracy(y_test, predictions))
