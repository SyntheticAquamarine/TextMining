import pandas as pd
from boto import sns
from matplotlib import pyplot as plt
from nltk import FreqDist, DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import Cleaning

Positive = pd.read_csv(r'C:\Users\janja\Desktop\II_stopien\Kolokwium_Marek_Jedrychowski\Data\Positive.csv',
                       usecols=['text'])
Positive['Sentiment'] = 'Positive'
Neutral = pd.read_csv(r'C:\Users\janja\Desktop\II_stopien\Kolokwium_Marek_Jedrychowski\Data\Neutral.csv',
                      usecols=['text'])
Neutral['Sentiment'] = 'Neutral'
Negative = pd.read_csv(r'C:\Users\janja\Desktop\II_stopien\Kolokwium_Marek_Jedrychowski\Data\Negative.csv',
                       usecols=['text'])
Negative['Sentiment'] = 'Negative'

Concatenated = pd.concat([Positive, Neutral, Negative])

X = Concatenated['text']
Y = Concatenated['Sentiment']

X_tr, X_tst, y_tr, y_tst = train_test_split(X, Y, test_size=.20, random_state=0)

vectorization = CountVectorizer(tokenizer=Cleaning.text_tokenizer)
X_training = vectorization.fit_transform(X_tr)
X_testing = vectorization.transform(X_tst)
logreg = LogisticRegression(C=1e5)
y_pred = logreg.predict(Y)

def informations():
    text = Cleaning.process()
    # Creates a frequency distribution for
    frqdist = FreqDist(text)
    mostCommon = frqdist.most_common(200)
    print(mostCommon)


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

def dt():
    dd = DecisionTreeClassifier()
    dd = dd.fit(X_training, y_tr)
    dd_ocena = dd.predict(X_testing)
    print(f'Decission tree: \n {classification_report(y_tst, dd_ocena)}')
    heatmap(y_tst, dd_ocena, "Decission tree")
