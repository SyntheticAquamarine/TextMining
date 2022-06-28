import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import text_tokenizer, generate_dataframe, wordclouds, show_wordclouds, show_plots
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tabulate import tabulate


def token_weights(df: pd.DataFrame):
    """Show 10 highest token weights for positive reviews"""
    df_pos = df[df['rating'] > 2]
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform_pos = vectorizer.fit_transform(df_pos['verified_reviews'])
    column_names_pos = vectorizer.get_feature_names_out()
    array_pos = x_transform_pos.toarray()
    token_column_sums_pos = np.sum(array_pos, axis=0)
    highest_weight_indexes_pos = np.argpartition(token_column_sums_pos, -12)[-12:]
    highest_weight_token_names_pos = []
    highest_weight_pos = []

    for index in np.nditer(highest_weight_indexes_pos):
        highest_weight_token_names_pos.append((column_names_pos[index]))
        highest_weight_pos.append(token_column_sums_pos[index])

    data_pos = {'Tokens': highest_weight_token_names_pos, 'Weights': highest_weight_pos}
    tokens_pos = pd.DataFrame(data_pos)
    sorted_tokens = tokens_pos.sort_values(by=['Weights'], ascending=True)
    print(tabulate(sorted_tokens, headers='keys', tablefmt='psql'))
    plt.bar(sorted_tokens['Tokens'], sorted_tokens['Weights'])
    plt.xlabel("Tokens")
    plt.ylabel("Weights")
    plt.xticks(rotation=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title("10 most important tokens for positive reviews")
    plt.show()


def sentiment(df: pd.DataFrame):
    """Logistic Regression,
    RandomForestClassifier, Support Vector Machine"""
    x = df['verified_reviews']
    y = df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform_train = vectorizer.fit_transform(x_train)
    x_transform_test = vectorizer.transform(x_test)
    r_log = LogisticRegression()
    r_log.fit(x_transform_train, y_train)
    lr_score = r_log.score(x_transform_test, y_test)
    print(f"Logistic regression model prediction accuracy - {lr_score * 100} %")

    svml = svm.SVC()
    svml = svml.fit(x_transform_train, y_train)
    svml_score = svml.score(x_transform_test, y_test)
    print(f"Support Vector Machine model prediction accuracy - {svml_score * 100} %")
    y_pred_svml = svml.predict(x_transform_test)
    print("Classification report for Support Vector Machine")
    print(classification_report(y_test, y_pred_svml))
    rfcl = RandomForestClassifier(n_estimators=150)
    rfcl = rfcl.fit(x_transform_train, y_train)
    rfcl_score = rfcl.score(x_transform_test, y_test)
    print(f"Random forest classifier prediction accuracy = {rfcl_score * 100} %")
    y_pred_rfcl = rfcl.predict(x_transform_test)
    print("Classification report for Random Forest Classifier")
    print(classification_report(y_test, y_pred_rfcl))

def main():
    df = generate_dataframe()
    show_plots(df)
    wordclouds(df)
    show_wordclouds()
    token_weights(df)
    sentiment(df)


if __name__ == "__main__":
    main()
