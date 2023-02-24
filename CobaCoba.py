import pandas as pd
from sklearn import preprocessing
from sklearn import feature_extraction
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import statistics
from imblearn.over_sampling import SMOTE
import pickle
from mlxtend.evaluate import bias_variance_decomp

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = set(stopwords.words('indonesian'))

data = pd.read_excel('new_chat2.xlsx')

intent = data.groupby("Intent").filter(lambda x: len(x) >= 5)

dataframe = data[(data.Intent.isin(intent['Intent']))].reset_index(drop=True)

message = dataframe['Message']
y = dataframe['Intent']

encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
X = []
for kata in message:
    X.append(kata)

vectorize = feature_extraction.text.CountVectorizer(stop_words=stop, lowercase=True)
vector = feature_extraction.text.TfidfVectorizer(stop_words=stop, lowercase=True)
vector.fit(X)
# with open('training_data_tfidf_vectorize1','wb') as f:
#     pickle.dump(vector,f)
X = vector.transform(X)

# mxgb = XGBClassifier(use_label_encoder=False,objective="multi:softmax",num_class=13)
mxgb = XGBClassifier(use_label_encoder=False)

classification_report_XGBoost = []
cm_XGBoost = []
expected_loss = []
bias = []
var = []

kf = model_selection.StratifiedKFold(n_splits=5,random_state=None)
for train_index,test_index in kf.split(X,y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
    sm = SMOTE(k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train,y_train)
    mxgb.fit(X_res, y_res)
    # mxgb.fit(X_train, y_train)
    prediction = mxgb.predict(X_test)
    matrix = classification_report(y_test, prediction,output_dict=True, zero_division=0)
    classification_report_XGBoost.append(matrix)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(mxgb, X_res,
                                                                y_res, X_test,
                                                                y_test,
                                                                loss='mse',
                                                                num_rounds=50,
                                                                random_seed=20)
    expected_loss.append(avg_expected_loss)
    bias.append(avg_bias)
    var.append(avg_var)
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
print("")
print("Average Expected Loss: ", statistics.fmean(expected_loss))
print("Average Bias: ", statistics.fmean(bias))
print("Average Variance: ", statistics.fmean(var))
# with open('model_pickle_TF-IDF_SMOTE1','wb') as f:
#     pickle.dump(mxgb,f)
