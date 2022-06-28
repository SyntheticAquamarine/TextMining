import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Projekt.funkcje.cleaning import file_read, text_tokenizer


def imp_words(tst: str):
    csv_data = file_read(tst)
    reviews = csv_data['text']
    print("Most important tokens")

    # https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a

    tfid_vctr = TfidfVectorizer(tokenizer=text_tokenizer)
    tfid_tfrm = tfid_vctr.fit_transform(reviews)
    wdict_2 = tfid_vctr.get_feature_names_out(reviews)
    im_words = sum(tfid_tfrm.toarray())
    y = np.argpartition(im_words, -15)[-15:]
    plt.barh(wdict_2[y], im_words[y])
    plt.title('Most important words')
    plt.ylabel('Word')
    plt.xlabel('TFIDF')
    plt.show()
    col_1 = ["Word", "TF-IDF Value"]
    new_table_2 = PrettyTable()
    new_table_2.add_column(col_1[0], wdict_2[y])
    new_table_2.add_column(col_1[1], im_words[y])
    new_table_2.sortby = col_1[1]
    print(new_table_2)


def most_popular(tst: str):
    csv_data = file_read(tst)
    reviews = csv_data['text']
    print("Most popular tokens")
    c_vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform = c_vectorizer.fit_transform(reviews)
    wdict = c_vectorizer.get_feature_names_out(reviews)
    words = sum(x_transform.toarray())

    # argpartition - https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html

    x = np.argpartition(words, -15)[-15:]
    plt.barh(wdict[x], words[x])
    plt.title('Most used words')
    plt.ylabel('word')
    plt.xlabel('Number of uses')
    plt.show()
    col = ["word", "Number of uses"]
    word_table = PrettyTable()
    word_table.add_column(col[0], wdict[x])
    word_table.add_column(col[1], words[x])

    # sortowanie tabeli https://stackoverflow.com/questions/37423445/python-prettytable-sort-by-multiple-columns
    word_table.sortby = col[1]
    print(word_table)
