import pandas as pd
import numpy as np


def load_dataset(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Loads dataset

    Args:
        data_dir (str): Folder that contains train,test,with_embeddings csv files

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]: dataframe with embeddings, test dataframe with embeddings, labels
    """
    df = pd.read_csv(f"{data_dir}/with_embeddings.csv", index_col=None)
    df['embeddings'] = df["embeddings"].apply(
        lambda x: np.fromstring(x[1:-1], sep=","))  # type: ignore
    df_test = pd.read_csv(f"{data_dir}/test_with_embeddings.csv")
    df_test['embeddings'] = df_test["embeddings"].apply(
        lambda x: np.fromstring(x[1:-1], sep=","))  # type: ignore
    labels = 'toxic severe_toxic obscene threat insult identity_hate'.split()
    labels = df[labels].sum().sort_values(ascending=False).keys()
    return df, df_test, labels
