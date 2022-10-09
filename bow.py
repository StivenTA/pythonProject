import re
from cleantext import clean
from sklearn.feature_extraction.text import TfidfVectorizer


text = "                This sample text contains               laughing emojis 😀 😃 😄 😁 😆 😅 😂 🤣 https://datascienceparichay.com/article/python-check-if-tuple-is-empty/"

#print text after removing the emojis from it
# text = clean(text, no_emoji=True)
print(text.strip())

# pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))';
# match = re.findall(pattern, text)
# # print(match[0][0])
# print(match)
# if len(match) > 0:
#     for m in match:
#         url = m[0]
#         text = text.replace(url, '')
#         print(text)
# print(text)



# # assign documents
# d0 = 'Geeks for geeks'
# d1 = 'Geeks'
# d2 = 'r2j'
#
# # merge documents into a single corpus
# string = [d0, d1, d2]
#
# # create object
# tfidf = TfidfVectorizer()
#
# # get tf-df values
# result = tfidf.fit_transform(string)
#
# # get idf values
# print('\nidf values:')
# for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
#     print(ele1, ':', ele2)
#
# # get indexing
# print('\nWord indexes:')
# print(tfidf.vocabulary_)
#
# # display tf-idf values
# print('\ntf-idf value:')
# print(result)
#
# # in matrix form
# print('\ntf-idf values in matrix form:')
# print(result.toarray())