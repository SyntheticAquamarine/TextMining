import wordcloud as wc
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from Cleaning import process


def word_cloud():
    document = process()
    unique_values = list(set(document))
    bow = {u: document.count(u) for u in unique_values}
    wc = WordCloud(width=4000, height=3000, background_color='white', colormap='bone')
    cloud = wc.generate_from_frequencies(bow)
    plt.axis("off")
    plt.imshow(cloud, interpolation='bilinear')
    plt.show()
