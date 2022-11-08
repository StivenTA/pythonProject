import pandas as pd
from sklearn import preprocessing
from sklearn import feature_extraction
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import statistics
import math
from imblearn.over_sampling import SMOTE
import pickle

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = set(stopwords.words('indonesian'))

df = pd.read_excel('distance_dataset.xlsx')
data = pd.read_excel('Chat_Intent.xlsx')
# print(df)
# print(data['Intent'].value_counts())
intent = data.groupby("Intent").filter(lambda x: len(x) >= 5)
# print(intent)
dataframe = data[(data.Intent.isin(intent['Intent']))].reset_index(drop=True)
# print(dataframe)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
message = dataframe['Message']
y = dataframe['Intent']

encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
X = []
for kata in message:
    X.append(stemmer.stem(kata))

vectorize = feature_extraction.text.CountVectorizer(stop_words=stop, lowercase=True)
vector = feature_extraction.text.TfidfVectorizer(stop_words=stop, lowercase=True)
vectorize.fit(X)
X = vectorize.transform(X)

# # X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,random_state=4,stratify=y)

mxgb = XGBClassifier(use_label_encoder=False)
svm = svm.SVC()
mlp = MLPClassifier()
knn = KNeighborsClassifier(n_neighbors=3)
# mxgb.fit(X_train,y_train)
# prediction = mxgb.predict(X_test)
# matrix = classification_report(y_test,prediction)
# print(matrix)

classification_report_MLP = []
classification_report_SVM = []
classification_report_XGBoost = []
classification_report_KNN = []

kf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=None)
for train_index,test_index in kf.split(X,y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
    sm = SMOTE(k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train,y_train)
    # sm = SMOTE()
    # X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train,y_train)
    mxgb.fit(X_res, y_res)
    prediction = mxgb.predict(X_test)
    matrix = classification_report(y_test, prediction,output_dict=True, zero_division=0)
    classification_report_XGBoost.append(matrix)
    # classification_report_XGBoost.append(mxgb.score(X_test, y_test))

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sm = SMOTE(k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    svm.fit(X_res,y_res)
    svm_pred = svm.predict(X_test)
    report_svm = classification_report(y_test,svm_pred, output_dict=True, zero_division=0)
    classification_report_SVM.append(report_svm)
    # classification_report_SVM.append(svm.score(X_test, y_test))

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sm = SMOTE(k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    mlp.fit(X_res,y_res)
    mlp_pred = mlp.predict(X_test)
    report_mlp = classification_report(y_test,mlp_pred, output_dict=True, zero_division=0)
    classification_report_MLP.append(report_mlp)
    # classification_report_MLP.append(mlp.score(X_test,y_test))

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sm = SMOTE(k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    knn.fit(X_res,y_res)
    knn_pred = knn.predict(X_test)
    report_knn = classification_report(y_test,knn_pred, output_dict=True, zero_division=0)
    classification_report_KNN.append(report_knn)
    # classification_report_KNN.append(knn.score(X_test,y_test))



NewDict = newDict = {
    '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '9': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '10': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '11': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '12': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
}

for i in classification_report_XGBoost:
    for key,value in i.items():
        if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg' :
            # print(key,value.keys())
            if list(value.keys())[0] == 'precision':
                NewDict[key]['precision'].append(value['precision'])
            else:
                break
            if list(value.keys())[1] == 'recall':
                NewDict[key]['recall'].append(value['recall'])
            else:
                break
            if list(value.keys())[2] == 'f1-score':
                NewDict[key]['f1-score'].append(value['f1-score'])
            else:
                break
            if list(value.keys())[3] == 'support':
                NewDict[key]['support'].append(value['support'])
            else:
                break

for key,value in NewDict.items():
    print('Class: ', key, ' | Avg Precision: ', round(statistics.fmean(value['precision']),2), ' | Avg Recall: ', round(statistics.fmean(value['recall']),2), ' | Avg F1-Score: ', round(statistics.fmean(value['f1-score']),2), ' | Support: ', value['support'])

mean_xgboost = []
for i in classification_report_XGBoost:
    mean_xgboost.append(i['accuracy'])
print("Average Accuracy for XGBoost: ", round(statistics.fmean(mean_xgboost),3))
# print(statistics.fmean(precision_0))
print("")
NewDict = newDict = {
    '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '9': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '10': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '11': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '12': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
}

for i in classification_report_SVM:
    for key,value in i.items():
        if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg' :
            # print(key,value.keys())
            if list(value.keys())[0] == 'precision':
                NewDict[key]['precision'].append(value['precision'])
            else:
                break
            if list(value.keys())[1] == 'recall':
                NewDict[key]['recall'].append(value['recall'])
            else:
                break
            if list(value.keys())[2] == 'f1-score':
                NewDict[key]['f1-score'].append(value['f1-score'])
            else:
                break
            if list(value.keys())[3] == 'support':
                NewDict[key]['support'].append(value['support'])
            else:
                break

for key,value in NewDict.items():
    print('Class: ', key, ' | Avg Precision: ', round(statistics.fmean(value['precision']),2), ' | Avg Recall: ', round(statistics.fmean(value['recall']),2), ' | Avg F1-Score: ', round(statistics.fmean(value['f1-score']),2), ' | Support: ', value['support'])

mean_svm = []
for i in classification_report_SVM:
    mean_svm.append(i['accuracy'])
print("Accuracy of SVM: ",statistics.mean(mean_svm))
print("")
NewDict = newDict = {
    '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '9': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '10': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '11': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '12': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
}

for i in classification_report_MLP:
    for key,value in i.items():
        if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg' :
            # print(key,value.keys())
            if list(value.keys())[0] == 'precision':
                NewDict[key]['precision'].append(value['precision'])
            else:
                break
            if list(value.keys())[1] == 'recall':
                NewDict[key]['recall'].append(value['recall'])
            else:
                break
            if list(value.keys())[2] == 'f1-score':
                NewDict[key]['f1-score'].append(value['f1-score'])
            else:
                break
            if list(value.keys())[3] == 'support':
                NewDict[key]['support'].append(value['support'])
            else:
                break

for key,value in NewDict.items():
    print('Class: ', key, ' | Avg Precision: ', round(statistics.fmean(value['precision']),2), ' | Avg Recall: ', round(statistics.fmean(value['recall']),2), ' | Avg F1-Score: ', round(statistics.fmean(value['f1-score']),2), ' | Support: ', value['support'])

mean_mlp = []
for i in classification_report_MLP:
    mean_mlp.append(i['accuracy'])
print("Accuracy of MLP: ",statistics.mean(mean_mlp))
print("")
NewDict = newDict = {
    '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '9': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '10': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '11': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '12': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
}

for i in classification_report_KNN:
    for key,value in i.items():
        if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg' :
            # print(key,value.keys())
            if list(value.keys())[0] == 'precision':
                NewDict[key]['precision'].append(value['precision'])
            else:
                break
            if list(value.keys())[1] == 'recall':
                NewDict[key]['recall'].append(value['recall'])
            else:
                break
            if list(value.keys())[2] == 'f1-score':
                NewDict[key]['f1-score'].append(value['f1-score'])
            else:
                break
            if list(value.keys())[3] == 'support':
                NewDict[key]['support'].append(value['support'])
            else:
                break

for key,value in NewDict.items():
    print('Class: ', key, ' | Avg Precision: ', round(statistics.fmean(value['precision']),2), ' | Avg Recall: ', round(statistics.fmean(value['recall']),2), ' | Avg F1-Score: ', round(statistics.fmean(value['f1-score']),2), ' | Support: ', value['support'])

mean_knn = []
for i in classification_report_KNN:
    mean_knn.append(i['accuracy'])
print("Accuracy of KNN: ",statistics.mean(mean_knn))

with open('model_pickle','wb') as f:
    pickle.dump(mxgb,f)
