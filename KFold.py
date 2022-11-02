from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neural_network import MLPClassifier
# from pycaret.classification import *
#

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1])
le = LabelEncoder
df = pd.read_excel("distance_dataset.xlsx")
# print(df)
array = df.values
X = array[:,0:1]
y = array[:,1]
y = np.int_(y)
# print(X.shape)
# print(type(y[0]))
# print(y)

skf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
# print(skf)
# skf.get_n_splits(x,y)
# print(skf)
model = xgb.XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
svm_model = svm.SVC()
mlp_model = MLPClassifier(random_state=5)
data_dmatrix = xgb.DMatrix(data=X,label=y)
# declare parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators': 100
}

# instantiate the classifier
xgb_clf = xgb.XGBClassifier(**params)
# # print(skf)
classification_report_MLP = []
classification_report_SVM = []
classification_report_XGBoost = []
for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = le().fit_transform(y[train_index]), y[test_index]
    # print("Y Train: ", set(y_train),"Y Test: ", set(y_test))
    # print(y_train)
    xgb_clf.fit(X_train,y_train)
    prediction = xgb_clf.predict(X_test)
    predictions = [round(value) for value in prediction]
    report = confusion_matrix(y_test,predictions)
    print(report)
    classification_report_XGBoost.append(report)

    svm_model.fit(X_train,y_train)
    svm_prediction = svm_model.predict(X_test)
    report_SVM = classification_report(y_test,svm_prediction)
    # print(report_SVM)
    # print(classification_report(y_test,svm_prediction))
    classification_report_SVM.append(report_SVM)

    mlp_model.fit(X_train,y_train)
    mlp_prediction = mlp_model.predict(X_test)
    report_mlp = classification_report(y_test,mlp_prediction)
    # print(report_mlp)
    classification_report_MLP.append(report_mlp)
# print(classification_report_XGBoost)
# for i in accuracy:
#     print(i)
# params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}
#
# xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
# print(xgb_cv)