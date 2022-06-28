from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from .cleaning import *

data = file_read('./Data/tweets_airline.csv')
# Dzielenie zbioru na treningowy i testowy przy pomocy funkcji train_test_split
# predict https://www.askpython.com/python/examples/python-predict-function

X_tr, X_tst, y_tr, y_tst = train_test_split(data['text'], data['airline_sentiment'],
                                            test_size=0.20, shuffle=True, stratify=data['airline_sentiment'])
i = 0
X_tr = X_tr.dropna()

vectorization = CountVectorizer(tokenizer=text_tokenizer)
X_training = vectorization.fit_transform(X_tr)
X_testing = vectorization.transform(X_tst)


def heatmap(y_test, y_pred, classifier):
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, cmap="RdBu_r", fmt='g', cbar=False),
    ax.xaxis.set_ticklabels(['Positive', 'Neutral', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Neutral', 'Negative'])
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(f'{classifier} Confusion Matrix')
    plt.show()


# BaggingClassifier
def bagg():
    bc_model = BaggingClassifier()
    bc_model.fit(X_training, y_tr)
    bc_predykcja = bc_model.predict(X_testing)
    print(f'BaggingClassifier \n {classification_report(y_tst, bc_predykcja)}')
    heatmap(y_tst, bc_predykcja, "BaggingClassifier")


# RandomForestClassifier
def rfc():
    rfc = RandomForestClassifier()
    rfc.fit(X_training, y_tr)
    rfc_predykcja = rfc.predict(X_testing)
    print(f'RandomForestClassifier \n {classification_report(y_tst, rfc_predykcja)}')
    heatmap(y_tst, rfc_predykcja, "RandomForestClassifier")


# Support Vector Classifier
def svm():
    svm = SVC()
    svm.fit(X_training, y_tr)
    svm_ocena = svm.predict(X_testing)
    print(f'SVM  \n {classification_report(y_tst, svm_ocena)}')
    heatmap(y_tst, svm_ocena, "SVM")


# Decission tree
def dt():
    dd = DecisionTreeClassifier()
    dd = dd.fit(X_training, y_tr)
    dd_ocena = dd.predict(X_testing)
    print(f'Decission tree: \n {classification_report(y_tst, dd_ocena)}')
    heatmap(y_tst, dd_ocena, "Decission tree")


# AdaBoostClassifier
def ab():
    abcfr = AdaBoostClassifier()
    abcfr = abcfr.fit(X_training, y_tr)
    ab_ocena = abcfr.predict(X_testing)
    print(f'AdaBoostClassifier: \n {classification_report(y_tst, ab_ocena)}')
    heatmap(y_tst, ab_ocena, "AdaBoostClassifier")


# Logistic regression
def reg_log():
    reg_l = LogisticRegression(max_iter=4500)
    reg_l.fit(X_training, y_tr)
    reg_l_przew = reg_l.predict(X_testing)
    print(f'Logistic regression \n {classification_report(y_tst, reg_l_przew)}')
    heatmap(y_tst, reg_l_przew, "Logistic regression")
