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


def IDF(documents):
    import math
    N = len(documents)
    print(N)
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


# idfs = IDF([numOfWordsA,numOfWordsB])
# doc = "In the expression named entity, the word named restricts the task to those entities for which one or many strings, such as words or phrases, stands (fairly) consistently for some referent. This is closely related to rigid designators, as defined by Kripke,[3][4] although in practice NER deals with many names and referents that are not philosophically 'rigid'. For instance, the automotive company created by Henry Ford in 1903 can be referred to as Ford or Ford Motor Company, although 'Ford' can refer to many other entities as well (see Ford). Rigid designators include proper names as well as terms for certain biological species and substances,[5] but exclude pronouns (such as 'it'; see coreference resolution), descriptions that pick out a referent by its properties (see also De dicto and de re), and names for kinds of things as opposed to individuals (for example 'Bank')."
# count_vec = CountVectorizer()
# count_occurs = count_vec.fit_transform([doc])
#
# count_occurs_df = pd.DataFrame(
#     (count, word) for word, count in zip(count_occurs.toarray().tolist()[0],count_vec.get_feature_names_out())
# )
# count_occurs_df.columns = ['Word','Count']
# count_occurs_df.sort_values('Count',ascending=False,inplace=True)
# print(count_occurs_df)

A = "The rain is pouring the"
B = "The rain has stopped"
C = "The sun is up The sun is set The sun is round Rain and sun are the best"
Dataset = {
    'data': ['Why sun is up at east and set at west?',
             'There is only one sun and one moon at earth',
             'i love sunny day',
             'why people always skeptical when see sun is up at west, thats what i thought',
             'Man this life is suck'],
    'target': [1, 2, 2, 1, 2]
}
A = A.lower()
# B = B.lower()
# C = C.lower()
bowDataset = []
for i in Dataset['data']:
    bowDataset.append(i.lower().split(' '))
# B = B.lower()
print(bowDataset)


# bowA = A.split(' ')
# bowB = B.split(' ')
# bowC = C.split(' ')
# bowC = []
# for i in C:
#     bowC.append((i.lower().split(' ')))
bowDataset = np.array(list(bowDataset), dtype=object)
# print (bowDataset)
# print(bowC)
# print(bowA)
# print(bowB)

# uniqueWords = set(bowA).union(set(bowC).union(set(bowB)))
# print(uniqueWords)
uniqueWords = set()
for sentence in bowDataset:
    uniqueWords = uniqueWords.union(set(sentence))
print(uniqueWords)

# numOfWordsA = dict.fromkeys(uniqueWords, 0)
# for word in bowA:
#     numOfWordsA[word] += 1
# # numOfWordsB = dict.fromkeys(uniqueWords,0)
# # for word in bowB:
# #     numOfWordsB[word] += 1
# # numOfWordsC = dict.fromkeys(uniqueWords,0)
# # for word in bowC:
# #     numOfWordsC[word] += 1
numOfWordsDataset = dict.fromkeys(uniqueWords, 0)
for sentence in bowDataset:
    for word in sentence:
        numOfWordsDataset[word] += 1
# # print(numOfWordsA)
# # print(numOfWordsB)
# # print(numOfWordsC)
print(numOfWordsDataset)
# # #
# tfA = TF(numOfWordsA, bowA)
# # tfB = TF(numOfWordsB,bowB)
# # tfC = TF(numOfWordsC,bowC)
# tfDataset = TF(numOfWordsDataset, bowDataset)
# # print(tfDataset)
# #
# # print(tfC)
# #
# # print(tfA)
# # #print(tfB)
# #
# idfs = IDF([numOfWordsDataset, numOfWordsA])
# print(idfs)
# #
# tfidfA = TFIDF(tfA, idfs)
# # tfidfB = TFIDF(tfB,idfs)
# # tfidfC = TFIDF(tfC,idfs)
# tfidfDataset = {'Data': TFIDF(tfDataset, idfs)}
#
# print(tfidfA)
# # print(tfidfB)
# # print(tfidfC)
# print(tfidfDataset)
#
# value = sum(tfidfA.values())
# print(value)
#
# df = pd.DataFrame(tfidfDataset)
# # df = pd.DataFrame([tfidfDataset])
# print(df.shape)
#
# X, y = df.values, df.values
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234
# )
#
# k = 3
# clf = KNN(k=k)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("KNN classification accuracy", accuracy(y_test, predictions))
