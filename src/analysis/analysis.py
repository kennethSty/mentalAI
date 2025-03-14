import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random
import gc

def publication_dates():
    files = [
        "data/00_raw/pubmed/pubmed_abstracts2019.csv",
        "data/00_raw/pubmed/pubmed_abstracts2020.csv",
        "data/00_raw/pubmed/pubmed_abstracts2021.csv",
        "data/00_raw/pubmed/pubmed_abstracts2022.csv",
        "data/00_raw/pubmed/pubmed_abstracts2023.csv",
        "data/00_raw/pubmed/pubmed_abstracts20172018.csv",
    ]

    num_articles = {}

    for filename in files:
        print(filename)
        with open(filename, "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["year"] in num_articles:
                    num_articles[row["year"]] += 1
                else:
                    num_articles[row["year"]] = 1

    num_articles = collections.OrderedDict(sorted(num_articles.items()))
    articles = num_articles.copy()

    for key, value in num_articles.items():
      if value <= 100:
        articles.pop(key, None)

    print(articles)

    plt.figure(figsize=(10, 6))
    plt.bar(articles.keys() , articles.values())
    plt.xlabel("Publication Year")
    plt.ylabel("Number of Articles")
    plt.title("Number of Articles by Publication Date")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plot_path = "plots/num_articles_pubmed.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Histogram saved at: {plot_path}")

def ratio_human_synthetic():
    human_convs = 0
    synthetic_convs = 0

    with open("data/02_train_test_splits/test/counsel_conversations_test.csv", 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["origin"] == "human":
                human_convs += 1
            elif row["origin"] == "synthetic":
                synthetic_convs += 1

    with open("data/02_train_test_splits/train/counsel_conversations_train.csv", 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["origin"] == "human":
                human_convs += 1
            elif row["origin"] == "synthetic":
                synthetic_convs += 1

    print(f"Human Conversations: {human_convs}")
    print(f"Synthetic Conversations: {synthetic_convs}")
    print(f"Ratio (Human/Synthetic): {human_convs / synthetic_convs}")

def wordcloud(files, col_name, out):
    words = []
    for filename in files:
        print(filename)
        df = pd.read_csv(filename, index_col=None)

        for abstract in df[col_name]:
            for word in str(abstract).split():
                words.append(word)

    print (f"There are {len(words)} words in all of the abstracts")

    subset_words = random.sample(words, 100000)
    del words
    gc.collect()

    print("Selected Subset")

    subset_text = " ".join(subset_words)
    print("Joined")

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(subset_text)
    wordcloud.to_file(out)

if __name__ == "__main__":
    wordcloud(files=[
        "data/00_raw/pubmed/pubmed_abstracts2019.csv",
        "data/00_raw/pubmed/pubmed_abstracts2020.csv",
        "data/00_raw/pubmed/pubmed_abstracts2021.csv",
        "data/00_raw/pubmed/pubmed_abstracts2022.csv",
        "data/00_raw/pubmed/pubmed_abstracts2023.csv",
        "data/00_raw/pubmed/pubmed_abstracts20172018.csv",
    ], col_name="abstract", out="plots/wordcloud_pubmed.png")

    wordcloud(files=[
        "data/02_train_test_splits/test/counsel_conversations_test.csv",
        "data/02_train_test_splits/train/counsel_conversations_train.csv",
    ], col_name="question_answer_pair(s)", out="plots/wordcloud_conv.png")

    publication_dates()

    ratio_human_synthetic()