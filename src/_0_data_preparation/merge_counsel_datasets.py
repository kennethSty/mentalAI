import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

def merge_counsel_datasets(paths_to_human_datasets: List[str], paths_to_synthetic_datasets: List[str]):
    human_df = pd.concat([pd.read_csv(path) for path in paths_to_human_datasets])
    human_df["origin"] = human_df.apply(lambda row: "human", axis=1)
    human_train_df, human_test_df = train_test_split(human_df, test_size=0.1, random_state=42)

    synthetic_df = pd.concat([pd.read_csv(path) for path in paths_to_synthetic_datasets])
    synthetic_df["origin"] = synthetic_df.apply(lambda row: "synthetic", axis=1)
    synthetic_train_df, synthetic_test_df = train_test_split(synthetic_df, test_size=0.1, random_state=42)

    train_df = pd.concat([human_train_df, synthetic_train_df])
    test_df = pd.concat([human_test_df, synthetic_test_df])
    train_df.to_csv("../../data/02_merged/train/counsel_conversations_train.csv", index=False)
    test_df.to_csv("../../data/02_merged/test/counsel_conversations_test.csv", index=False)

if __name__ == "__main__":

    paths_to_human_datasets = [
        "../../data/01_preprocessed/counsel_chat_preproc.csv",
        "../../data/01_preprocessed/mental_conv_preproc.csv",
        "../../data/01_preprocessed/mental_faq_preproc.csv",
        "../../data/01_preprocessed/pair_data_preproc.csv",
    ]

    paths_to_synthetic_datasets = [
        "../../data/01_preprocessed/synthetic_conversations_preproc.csv"
    ]

    merge_counsel_datasets(paths_to_human_datasets, paths_to_synthetic_datasets)
