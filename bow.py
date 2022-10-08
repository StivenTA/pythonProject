from sklearn.feature_extraction.text import TfidfVectorizer
import string

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
tfidf = TfidfVectorizer()

bowDataset = {'data': []}


# print(Dataset])
result = tfidf.fit_transform(Dataset['data'])
print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
    print(ele1, ':', ele2)

print(tfidf.vocabulary_)

print('\ntf-idf value:')
print(result)

# in matrix form
print('\ntf-idf values in matrix form:')
print(result.toarray())


