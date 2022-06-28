import re

from nltk import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd


def file_read(tst: str):
    tst = pd.read_csv(tst, usecols=['text', 'airline_sentiment'], encoding='UTF-8')
    return tst


def data_info(tst: str):
    tst = pd.read_csv(tst, sep=",", encoding='UTF-8')
    print(tst.head())
    print(tst.info())


def read_all(tst: str):
    tst = pd.read_csv(tst, sep=",", encoding='UTF-8')
    return tst


def text_cleaner(tst: str) -> str:
    res_ex3 = re.findall(r'[:;]+-?[>)(<]', tst)
    str = ""
    for s in tst:
        str += s
    emoji = ""
    for e in res_ex3:
        emoji += e
    emotes = re.sub('[:;]+-?[>)(<]', '', str)
    letters = emotes.lower()
    numbers = re.sub('\d+', '', letters)
    htm = re.sub('<.{1,9}>', '', numbers)
    pun = re.sub('[,.;:"]|(&#\d+;)', '', htm)
    pun = pun.strip()
    clean = pun + emoji
    return clean


def stemming(tst: str) -> list:
    stemmer = SnowballStemmer('english')
    stem = [stemmer.stem(word) for word in tst.split(' ')]
    return stem


def stop_words(tst: list) -> list:
    words = [word for word in tst if word not in stopwords.words('english')]
    return words


def text_tokenizer(tst: str) -> list:
    tst = stemming(tst)
    tst = stop_words(tst)
    tokenize = [word for word in tst if len(word) > 3]
    return tokenize


'''
def text_tokenizer(tst: str) -> list:
    stemmer = SnowballStemmer('english')
    tokenize = [stemmer.stem(word) for word in tst.split(' ') if
                word not in stopwords.words('english') and len(word) > 3]
    return tokenize
'''


def process_for_WC(tst: str) -> str:
    tst = text_cleaner(tst)
    tst = text_tokenizer(tst)
    return " ".join(tst)
