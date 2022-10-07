import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from SVM import SVM

# generating 2 class
X, y = datasets.make_blobs(
    n_samples=50,
    n_features=2,
    centers=2,
    cluster_std=1.05,
    random_state=40
)
y = np.where(y == 0, -1, 1)

# Call the SVM Algorithm
clf = SVM()

clf.fit(X, y)

print(clf.w, clf.b)


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    #garis tengah
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    #garis kiri (dekat warna ungu/hitam)
    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    #garis kanan (dekat warna kuning)
    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    #display garis
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


visualize_svm()

tfDataset = {'data': []}
idfs = {'data': []}
tfidf = {'data': []}
for sentence in bowDataset['data']:
    numOfSentenceDataset = dict.fromkeys(sentence,0)
    for word in sentence:
        numOfSentenceDataset[word] += 1
    tfTemp = TF(numOfSentenceDataset,sentence)
    tfDataset['data'].append(tfTemp)

    idfsTemp = IDF([numOfWordsDataset, sentence])
    idfs['data'].append(idfsTemp)

    # tfidfTemp = TFIDF([tfidf,idfs])
    # tfidf['data'].append(tfidfTemp)

print(tfidf)
