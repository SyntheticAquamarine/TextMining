from sklearn.feature_extraction import stop_words

import ReadFile
import re
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd


def texts(tst: str) -> str:
    #res_ex3 = re.findall('[:;]+-?[>)(<]', tst)
    default_str = ""
    for s in tst:
        default_str += s
    emotes = re.sub('[:;]+-?[>)(<]', '', default_str)
    letters = emotes.lower()
    numbers = re.sub('\d+', '', letters)
    htm = re.sub('<.{1,9}>', '', numbers)
    pun = re.sub('[,.;:"]', '', htm)
    pun = pun.strip()
    clean = pun + default_str
    return clean

# stemming, tokenizacja, remove stop words
def text_tokenizer(tst: str) -> list:
    stemmer = SnowballStemmer('english')
    stopwords.words('english')
    tokenize = [stemmer.stem(word) for word in tst.split(' ') if word not in stop_words and len(word) > 3]
    return tokenize


# def remove_stop_words(tst: list) -> list:
#    stopwords.words('english')
#    stop_clean = [token for token in tst if token not in stopwords]
#    return stop_clean


def process() -> list:
    tst = ReadFile.read_text()
    tst = texts(tst)
    tst = text_tokenizer(tst)
    # tst = remove_stop_words(tst)
    return tst
