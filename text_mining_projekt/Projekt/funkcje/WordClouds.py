from matplotlib import pyplot as plt
from wordcloud import WordCloud
from .cleaning import *


def wordclouds(tst: str):
    data_wordcloud = file_read(tst)
    data_wordcloud['text'] = data_wordcloud['text'].apply(process_for_WC)
    print(data_wordcloud['text'])
    sentiments = ['negative', 'neutral', 'positive']
    for sentiment in sentiments:
        data = data_wordcloud.query('airline_sentiment == @sentiment')['text'].str.cat(sep=' ')
        unique = list(set(list(data.split(" "))))
        bow = {u: data.count(u) for u in unique}

        wc = WordCloud(width=4000, height=3000, background_color='white', colormap='bone')
        cloud = wc.generate_from_frequencies(bow)
        plt.axis("off")
        plt.imshow(cloud, interpolation='bilinear')
        plt.title(f"Most frequent words in {sentiment} tweets")
        plt.show()
