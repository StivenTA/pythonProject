import re
from cleantext import clean
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# text = "                This sample text contains               laughing emojis 😀 😃 😄 😁 😆 😅 😂 🤣 https://datascienceparichay.com/article/python-check-if-tuple-is-empty/"
Dataset = {
    'data': ['Saya siapa? bagaimana dengan kabar dari apa yang mau saya ucapakan ya kan tidak saya tau',
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
# stop = set(stopwords.words('indonesian'))
# words = {'data':[]}
# for data in Dataset['data']:
#     words['data'].append(word_tokenize(data))
#
# # print(words['data'])
# clear_data = {'data':[]}
# for sentence in words['data']:
#     flag = 1
#     for word in sentence:
#         if word not in stop:
#             flag = 0
#     if(flag == 0):
#         clear_data['data'].append(sentence)
# print(clear_data)
test = 'Saya siapa? bagaimana Perekonomian mangajar dengan kabar dari apa yang mau saya ucapakan ya kan tidak saya tau'
text = ['Saya siapa? bagaimana diajarkan dengan kabar dari apa yang mau saya ucapakan ya kan tidak saya tau',
             'siapa dosen yang akan mengajar pada semester ini?',
             'i love sunny day',
             'why people always skeptical when see sun is up at west, thats what i thought',
             'Man this life is suck',
             'The sun is up The sun is set The sun is round Rain and sun are the best',
             'The rain has stopped',
             'The rain is pouring the']
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stem_word = {'data':[]}
for sentence in Dataset['data']:
    stem_word['data'].append(stemmer.stem(sentence))
print(stem_word)
tokenize_words = {'data':[]}
for word in stem_word['data']:
    tokenize_words['data'].append(word_tokenize(word))
stop = set(stopwords.words('indonesian'))
print(tokenize_words)
clear_Data = {'data':[]}
for sentence in tokenize_words['data']:
    temp_sentence = {'data':[]}
    for word in sentence:
        if word not in stop:
            temp_sentence['data'].append(word)
    clear_Data['data'].append(temp_sentence['data'])
# print(clear_Data)

