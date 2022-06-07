import pandas as pd


def divide_file():
    airlines = pd.read_csv(r'C:\Users\janja\Desktop\II_stopien\Kolokwium_Marek_Jedrychowski\Data\tweets_airline.csv',
                           sep=',', encoding='UTF-8')

    positive = airlines[airlines['airline_sentiment'] == 'positive']
    neutral = airlines[airlines['airline_sentiment'] == 'neutral']
    negative = airlines[airlines['airline_sentiment'] == 'negative']

    positive.to_csv('./Data/Positive.csv')
    negative.to_csv('./Data/Negative.csv')
    neutral.to_csv('./Data/Neutral.csv')


#divide_file()


def read_text():
    tst = pd.read_csv(r'C:\Users\janja\Desktop\II_stopien\Kolokwium_Marek_Jedrychowski\Data\tweets_airline.csv',
                      usecols=['text'], sep=",", encoding='UTF-8')
    return tst
