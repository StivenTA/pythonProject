# Standard Python Libaries
import string
import urllib.request
import os
import re
# Third Party Modules
import matplotlib.pyplot as plt
import pandas as pd # pip install pandas openpyxl
from cleantext import clean
import math
import statistics
import nltk
import spacy

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


data = []
with open(r"./WhatsApp_Chat_with_Pop_Aga_13yo_Cbb.txt",encoding="utf-8",mode='r') as f:
    data.append(f.readlines())
# with open(r"./WhatsApp_Chat_with_Andini__Amanda.txt",encoding="utf-8",mode='r') as f:
#     data.append(f.readlines())
# with open(r"./WhatsApp_Chat_with_Rian_davi_10yo.txt",encoding="utf-8",mode='r') as f:
#     data.append(f.readlines())
# with open(r"./WhatsApp_Chat_with_Billy_Toby_10__AGI_7.txt",encoding="utf-8",mode='r') as f:
#     data.append(f.readlines())
# with open(r"./WhatsApp_Chat_with_Garin__Abiyasa_14_Yo_CBB.txt",encoding="utf-8",mode='r') as f:
#     data.append(f.readlines())
# print(data[0][1])
exportdata = {'data': pd.DataFrame(columns=['Date', 'Time', 'Name', 'Message'])}
# exportdata = {'data': pd.DataFrame(columns=['Message'])}
# for i in data:
#     print(len(i))
for conversation in data:
    cleaned_data = []
    for line in conversation:
        # Check, whether it is a new line or not
        # If the following characters are in the line -> assumption it is NOT a new line
        # print(line)
        if '/' in line and ',' in line and ':' in line and '-' in line:
            # grab the info and cut it out
            date = line.split(",")[0]
            # print(date)
            line2 = line[len(date):]
            time = line2.split("-")[0][2:]
            # print(time)
            line3 = line2[len(time):]
            name = line3.split(":")[0][4:]
            # print(name)
            line4 = line3[len(name):]
            message = line4[6:-1]  # strip newline character
            # print(message)
            cleaned_data.append([date, time, name, message])
            # cleaned_data.append([message])
        # else, assumption -> new line. Append new line to previous 'message'
        else:
            new = cleaned_data[-1][-1] + " " + line
            cleaned_data[-1][-1] = new
    df = pd.DataFrame(cleaned_data, columns = ['Date', 'Time', 'Name', 'Message'])
    # df = pd.DataFrame(cleaned_data, columns=['Message'])

    frames = [exportdata['data'],df]
    # frames = [exportdata, df]
    exportdata['data'] = pd.concat(frames,sort=False)
    # exportdata = pd.concat(frames, sort=False)
# print(len(exportdata['data']))
exportdata['data'].to_excel('NewChat.xlsx', index=False)
#
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# DatasetDict = {'data': exportdata['data'].values}
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# Tokenizing = {'data': []}
# stem_word = {'data':[]}
# # print(DatasetDict['data'])
# # for sentence in DatasetDict['data']:
# #     # print(stemmer.stem(sentence))
# #     for word in sentence:
# #         print(stemmer.stem(word))
#     # print(sentence)
# pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))';
#
# # for sentence in DatasetDict['data']:
# #     stem_word['data'].append(stemmer.stem(sentence))
# #
#
# for sentence in DatasetDict['data']:
#     for line in sentence:
#         text = line
#         # clean(text, no_emoji=True)
#         # menghilangkan data url
#         match = re.findall(pattern, line)
#         if (len(match) > 0):
#             for m in match:
#                 url = m[0]
#                 text = text.replace(url,'')
#         # print(text)
#         text = stemmer.stem(text.replace('<Media omitted>',''))
#         # print(text)
#         if text != '':
#             # Tokenizing['data'].append(
#             #     text
#             #         # menghilangkan kata <Media omitted> yang berarti berisikan media
#             #         .replace('<Media omitted>','')
#             #         # .replace(match[0][0],'')
#             #
#             #         # menghilangkan enter
#             #         .replace('\n', '')
#             #
#             #         # menghilangkan punctuation
#             #         .translate(str.maketrans('','',string.punctuation))
#             #
#             #         # menghilangkan spasi yang berlebih pada awal dan akhir kalimat
#             #         .strip()
#             #
#             #         # semua kata menjadi huruf kecil
#             #         .lower()
#             #
#             #         # memisahkan data berdasarkan spasi
#             #         .split(' ')
#             # )
#             Tokenizing['data'].append(word_tokenize(text))
# # print(Tokenizing['data'])
# clear_Data = {'data':[]}
# stop = set(stopwords.words('indonesian'))
# for sentence in Tokenizing['data']:
#     temp_sentence = {'data':[]}
#     for word in sentence:
#         if word not in stop and word != '':
#             temp_sentence['data'].append(word)
#     clear_Data['data'].append(temp_sentence['data'])
# # print(clear_Data)
# uniqueWords = set()
# for words in clear_Data['data']:
#     text = words
#     # Membuat dataset yang berisikan hanya kata yang tidak redundan
#     uniqueWords = uniqueWords.union(set(text))
# # uniqueWords.remove('')
# numOfWordsDataset = dict.fromkeys(uniqueWords, 0)
#
# for sentence in clear_Data['data']:
#     for word in set(sentence):
#         # mencari jumlah setiap kata muncul dari seluruh dataset
#         numOfWordsDataset[word] += 1
#
# tfDataset = {'data': []}
# for sentence in clear_Data['data']:
#     numOfSentenceDataset = dict.fromkeys(sentence,0)
#     for word in sentence:
#         # mencari jumlah kata muncul dari sebuah kalimat
#         numOfSentenceDataset[word] += 1
#     # Menghitung nilai Term Frequency untuk setiap kalimat
#     tfTemp = TF(numOfSentenceDataset,sentence)
#     tfDataset['data'].append(tfTemp)
# tfidf = {'data': []}
# idf = {'data': []}
#
# for sentence in tfDataset['data']:
#     # mencari nilai IDF pada setiap kalimat dataset
#     # len diambil untuk menghitung total dari kalimat yang ada
#     # IDF menggunakan numOfWordsDataset karena menghitung setiap kata pada kalimat pada seluruh kata unik yang ada
#     idfsTemp = IDF([numOfWordsDataset],len(exportdata['data']))
#
#     idf['data'].append(idfsTemp)
#
# for sentence in tfDataset['data']:
#     tfidfTemp = TFIDF(sentence, idf['data'])
#     tfidf['data'].append(tfidfTemp)
#
# value = {'data': []}
# for sentence in tfidf['data']:
#     # nilai tfidf diubah menjadi nilai rata-rata
#     value['data'].append(statistics.mean(sentence.values()))
# # print(value)
