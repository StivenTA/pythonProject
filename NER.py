import pandas as pd
from sklearn import preprocessing
from sklearn import feature_extraction
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report


df = pd.read_excel('distance_dataset.xlsx')
data = pd.read_excel('Chat_Intent.xlsx')
# print(df)
# print(data['Intent'].value_counts())
intent = data.groupby("Intent").filter(lambda x: len(x) >= 5)
dataframe = data[(data.Intent.isin(intent['Intent']))].reset_index(drop=True)
# print(dataframe)
X = dataframe['Message']
y = dataframe['Intent']

encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)

vectorize = feature_extraction.text.CountVectorizer()
vectorize.fit(X)
X = vectorize.transform(X)
# print(X)

# X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,random_state=4,stratify=y)

mxgb = XGBClassifier(use_label_encoder=False)
# mxgb.fit(X_train,y_train)
# prediction = mxgb.predict(X_test)
# matrix = classification_report(y_test,prediction)
# print(matrix)

kf = model_selection.StratifiedKFold(n_splits=3,shuffle=False,random_state=None)
for train_index,test_index in kf.split(X,y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
    mxgb.fit(X_train, y_train)
    prediction = mxgb.predict(X_test)
    matrix = classification_report(y_test, prediction)
    print(matrix)
