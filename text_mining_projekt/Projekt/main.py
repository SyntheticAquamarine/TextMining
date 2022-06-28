import matplotlib.pyplot as plt
import seaborn as sns

from Projekt.funkcje import tokens as tk
from Projekt.funkcje.cleaning import data_info, read_all
from funkcje import Classification as clfc

from funkcje.WordClouds import wordclouds as wc

inform = read_all('./Data/tweets_airline.csv')
print(data_info('./Data/tweets_airline.csv'))
airliness = read_all('Data/tweets_airline.csv')
print(airliness['airline_sentiment'].value_counts())
sns.countplot(data=airliness, x="airline", hue="airline_sentiment", palette='magma')

plt.figure(figsize=(16, 8))
plt.title("Number of reviews")
inform['airline_sentiment'].value_counts().plot.bar(color='teal', legend=None)
plt.xlabel("Rating")
plt.ylabel("Number of occurences")
plt.show()

# word cloud
wc('./Data/tweets_airline.csv')
# Najczęściej występujące słowa
tk.most_popular('./Data/tweets_airline.csv')
# Najważniejsze słowa
tk.imp_words('./Data/tweets_airline.csv')

# Klasyfikatory

print("Classifiers")
# BaggingClassifier
clfc.bagg()

# RandomForestClassifier
clfc.rfc()

# Support Vector Machine
clfc.svm()

# Decission tree
clfc.dt()

# AdaBoostClassifier
clfc.ab()

# Logistic regression
clfc.reg_log()

'''
Opis klasyfikacji
Najbardziej dokładne wyniki uzyskane zostały przy pomocy regresji logistycznej, jak również metody SVM
Dla obywdu tych metod uzyskano 77% trafności.
Precission:
W wypadku klasyfikatora SVM: 

Kolumna precision
Odpowiada na pytanie, jaki procent wcześniej poczynionych predykcji był poprawny
Spośród recenzji, które model zakwalifikował jako pozytywne 75% jest takie w rzeczywistości
W wypadku recenzji neutralnych 62% odpowiedzi zapisanych jako neutralne faktycznie ma neutralny wydźwięk
Dla odpowiedzi negatywnych 81% odpowiedzi zakwalifikowanych jako negatywne faktycznie jest negatywna

Kolumna recall
Odpowiada, jaki procent pozytywnych opinii został uchwycony
Wśród odpowiedzi pozytywnych model poprawnie zdiagnozował 56% opinii
Wśród odpowiedzi neutralnych model poprawnie zdiagnozował 50% opinii
Wśród odpowiedzi nagatywnych model poprawnie zdiagnozował 92% opinii

Kolumna f1-score
Wyjaśnia, jaki procent predykcji był poprawny
Dla wartości pozytywnych algorytm posiada rozpoznawalność wartości na poziomie 64%
Wśród wartości neutralnych algorytm posiada rozpoznawalność na poziomie 56%
Dla odpowiedzi negatywnych algorytm posiada rozpoznawalność na poziomie 86%


Dla regresji logistycznej

Kolumna precision
Spośród recenzji, które model zakwalifikował jako pozytywne 71% jest takie w rzeczywistości
W wypadku recenzji neutralnych 61% odpowiedzi zapisanych jako neutralne faktycznie ma neutralny wydźwięk
Dla odpowiedzi negatywnych 84% odpowiedzi zakwalifikowanych jako negatywne faktycznie jest negatywna

Kolumna recall
Wśród odpowiedzi pozytywnych model poprawnie zdiagnozował 66% opinii
Wśród odpowiedzi neutralnych model poprawnie zdiagnozował 55% opinii
Wśród odpowiedzi nagatywnych model poprawnie zdiagnozował 89% opinii

Kolumna f1-score
Dla wartości pozytywnych algorytm posiada rozpoznawalność wartości na poziomie 86%
Wśród wartości neutralnych algorytm posiada rozpoznawalność na poziomie 58%
Dla odpowiedzi negatywnych algorytm posiada rozpoznawalność na poziomie 66%

'''
