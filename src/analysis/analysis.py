import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import collections
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random
import gc
import ast

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
    plt.savefig("plots/num_articles_pubmed.png", dpi=300, bbox_inches="tight")
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

def eval_from_logs(files):
    all_vals = []
    nums = []
    means = []
    stds = []
    errors = []

    for filename in files:
        print(f"Reading {filename}")

        values = []
        with open(filename) as file:
            for line in file:
                if line.startswith("Scores: "):
                    s = line[line.find("["):line.find("]") + 1]
                    values.append(ast.literal_eval(s))

        vals = np.asarray(values).flatten()

        num = vals.size
        mean = vals.mean()
        std = vals.std()
        err = std / np.sqrt(num)

        print(f"Number of scores {num}")
        print(f"Avg.: {mean}")
        print(f"Std.: {std}")
        print(f"Error: {err}")

        print(f"Min: {vals.min()}, Max: {vals.max()}")

        all_vals.append(vals)
        nums.append(num)
        means.append(mean)
        stds.append(std)
        errors.append(err)

    names = [
        "vanilla",
        "sentiment",
        "emotion",
    ]
    width = 0.3
    plt.xticks(np.arange(3), names)
    plt.ylim(0.35, 0.55)
    plt.grid(color='#95a5a6', linestyle=':', axis='y', alpha=0.5)

    # plt.bar("vanilla", means[0], width=width)
    plt.bar(1 + np.arange(2) - width/2, [means[1], means[3]], width=width, label="topk=0") # no rag
    plt.bar(1 + np.arange(2) + width/2, [means[2], means[4]], width=width, label="topk=1") # with rag

    baseline = plt.axhline(y=means[0],linewidth=1, color='r', label="baseline")
    plt.legend()

    # plt.errorbar("vanilla",   means[0], yerr=errors[0], capsize=4, color='#555')
    plt.errorbar(1 - width/2, means[1], yerr=errors[1], capsize=4, color='#555')
    plt.errorbar(1 + width/2, means[2], yerr=errors[2], capsize=4, color='#555')
    plt.errorbar(2 - width/2, means[3], yerr=errors[3], capsize=4, color='#555')
    plt.errorbar(2 + width/2, means[4], yerr=errors[4], capsize=4, color='#555')
    
    plt.ylabel("Bleurt Score")
    plt.title("Mean Bleurt Scores")
    plt.savefig("plots/mean_bleurt_scores.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axs = plt.subplots(1, len(all_vals), sharey=True, tight_layout=True)
    n_bins = 20

    for i, vals in enumerate(all_vals):
        axs[i].hist(vals, weights=np.ones(len(vals)) / len(vals), bins=n_bins)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Distribution of Bleurt Scores")

    axs[0].set_title("Baseline",      fontsize=8)
    axs[1].set_title("Sent.+ Topk=0", fontsize=8)
    axs[2].set_title("Sent.+ Topk=1", fontsize=8)
    axs[3].set_title("Emot.+ Topk=0", fontsize=8)
    axs[4].set_title("Emot.+ Topk=1", fontsize=8)

    plt.savefig(f"plots/bleurt_scores_dist.png", dpi=300, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    # wordcloud(files=[
    #     "data/00_raw/pubmed/pubmed_abstracts2019.csv",
    #     "data/00_raw/pubmed/pubmed_abstracts2020.csv",
    #     "data/00_raw/pubmed/pubmed_abstracts2021.csv",
    #     "data/00_raw/pubmed/pubmed_abstracts2022.csv",
    #     "data/00_raw/pubmed/pubmed_abstracts2023.csv",
    #     "data/00_raw/pubmed/pubmed_abstracts20172018.csv",
    # ], col_name="abstract", out="plots/wordcloud_pubmed.png")

    # wordcloud(files=[
    #     "data/02_train_test_splits/test/counsel_conversations_test.csv",
    #     "data/02_train_test_splits/train/counsel_conversations_train.csv",
    # ], col_name="question_answer_pair(s)", out="plots/wordcloud_conv.png")

    # publication_dates()
    # ratio_human_synthetic()
    eval_from_logs(files=[
        "evals/llama2.txt",
        "evals/llama2_sentiment+suicide.txt",
        "evals/llama2_rag1+sentiment+suicide.txt",
        "evals/llama2_emotion+suicide.txt",
        "evals/llama2_rag1+emotion+suicide.txt",
    ])