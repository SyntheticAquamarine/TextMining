import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import matplotlib.image

from nltk import SnowballStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords


def text_cleaner(text: str) -> str:
    res_ex3 = re.findall(r'[:;]+-?[>)(<]', text)
    str = ""
    for s in text:
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


def stemming(word: str) -> str:
    stemmer = SnowballStemmer('english')
    return stemmer.stem(word)


def delete_stop_words(text: str) -> list:
    words = [word for word in text if word.lower not in stopwords.words('english')]
    return words


def text_tokenizer(text: str) -> list:
    cleaned = text_cleaner(text)
    tokens = word_tokenize(cleaned)
    words = delete_stop_words(tokens)

    return [stemming(w) for w in words if len(w) > 3]


def add_labels(x, y):
    for i in range(1,len(x)+1):
        plt.text(i,y[i-1], y[i-1], ha="center", va="bottom")


def generate_dataframe() -> pd.DataFrame:
    df = pd.read_csv('data/alexa_reviews.csv', sep=';',
                     usecols=['rating', 'verified_reviews'], encoding='cp1252')
    df['verified_reviews'].replace(' ', np.NaN, inplace=True)  # replace empty reviews with NaN
    df.dropna(how='any', inplace=True)  # delete reviews that are Nan
    df["sentiment"] = df.rating.apply(
        lambda x: 0 if x in [1, 2] else 1)
    return df


def show_plots(df: pd.DataFrame):
    ratings = df['rating'].value_counts().sort_index()  # count distribution of review ratings
    plt.bar(ratings.index, ratings.values)
    plt.title(f"Distribution of review ratings")
    plt.xlabel("Review rating score")
    plt.ylabel("Number of reviews")
    add_labels(ratings.index, ratings.values)
    plt.show()

    sentiments = df['sentiment'].value_counts()  # count distribution of positive and negative sentiment among reviews
    plt.pie(sentiments.values, shadow=True, labels=["Positive", "Negative"], startangle=90, autopct='%1.1f%%',
            colors=["Blue", "Green"])
    plt.title("Distribution of sentiments")
    plt.show()


def wordclouds(df: pd.DataFrame):
    stop_list = set(stopwords.words('english'))
    text_general = " ".join(review for review in df.verified_reviews.astype(str))
    text_general = text_cleaner(text_general)
    wc = WordCloud(width=4000, height=3000, stopwords=stop_list, background_color='white', colormap='Blues')
    wc.generate(text_general)
    wc.to_file('wordclouds/wc_general.png')


    df_pos = df[df['rating'] >= 3]
    text_pos = " ".join(review for review in df_pos.verified_reviews.astype(str))
    text_pos = text_cleaner(text_pos)
    wc.generate(text_pos)
    wc.to_file('wordclouds/wc_positive.png')

    df_neg = df[df['rating'] < 3]
    text_neg = " ".join(review for review in df_neg.verified_reviews.astype(str))
    text_neg = text_cleaner(text_neg)
    wc.generate(text_neg)
    wc.to_file('wordclouds/wc_negative.png')


def show_wordclouds():
    for path in glob.glob('wordclouds/*'):
        plt.imshow(matplotlib.image.imread(path))
        plt.title(path.split('\\')[1])
        plt.axis("off")
        plt.show()
