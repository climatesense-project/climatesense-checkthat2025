import ast
from functools import partial
from typing import Any, Dict, List, Union

import pandas as pd
from datasets import Dataset
from plotnine import aes, geom_bar, geom_text, ggplot, labs, theme_light
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold


class DataLoader5FCV:
    def __init__(self, tokenizer, n_folds, seed=0):
        self.tokenizer = tokenizer
        self.n_folds = n_folds
        self.seed = seed
        self.data = self._read_data()
        self.cv_datasets, self.cv_test_dfs = self._create_datasets()

    def _read_data(self):
        data = pd.read_csv("ct_train_data.tsv", sep="\t")  # TODO Add cleanning and remove fixed info.
        data["labels"] = data["labels"].apply(lambda x: ast.literal_eval(x))

        return data

    def _create_datasets(self):
        def tokenize(examples):
            return self.tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")

        def filter_split(split_indices: List, example: Union[Dict, Any], indices: int) -> List[bool]:
            return [True if idx in split_indices else False for idx in indices]

        cv_datasets = []
        cv_test_dfs = []

        dataset = Dataset.from_pandas(self.data[["text", "labels"]])
        dataset = dataset.map(tokenize, batched=True, batch_size=32, remove_columns=["text"])

        kf = KFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        kf.get_n_splits(self.data)

        for _fold, (train_index, test_index) in enumerate(kf.split(self.data)):
            assert len(set(train_index).intersection(set(test_index))) == 0

            train_dataset = dataset.filter(
                partial(filter_split, train_index), with_indices=True, batched=True, keep_in_memory=True
            )
            test_dataset = dataset.filter(
                partial(filter_split, test_index), with_indices=True, batched=True, keep_in_memory=True
            )

            cv_datasets.append((train_dataset, test_dataset))
            cv_test_dfs.append(self.data.iloc[test_index])

        return cv_datasets, cv_test_dfs

    def get_datasets_for_fold(self, fold):
        return self.cv_datasets[fold]

    def get_test_df_for_fold(self, fold):
        return self.cv_test_dfs[fold]


def compute_metrics(data):
    """Compute evaluation metrics for multi-label classification.

    This function calculates accuracy, precision, recall, and F1-score
    for each category (cat1, cat2, cat3) and computes the macro F1-score
    across all categories.

    Args:
        data (pd.DataFrame): A DataFrame containing predictions and labels.
            The DataFrame should have columns named 'cat1_pred', 'cat2_pred',
            'cat3_pred' for predictions, and a 'labels' column containing
            lists of true labels for each category.

    Returns:
        Dict[str, float]: A dictionary containing the computed metrics:
            - "{category}_avg_acc": Accuracy for each category.
            - "{category}_avg_prec": Precision for each category.
            - "{category}_avg_rec": Recall for each category.
            - "{category}_avg_f1": F1-score for each category.
            - "macro_f1": Macro F1-score across all categories.
    """
    metrics = {}
    for i, _cat in enumerate(["cat1", "cat2", "cat3"]):
        preds = data[f"{_cat}_pred"].apply(lambda x: int(x) == 1)
        labels = data["labels"].apply(lambda x: int(float(x[i])) == 1)  # noqa: B023

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        metrics.update({f"{_cat}_avg_acc": acc, f"{_cat}_avg_prec": prec, f"{_cat}_avg_rec": rec, f"{_cat}_avg_f1": f1})

    preds = data[[c for c in data.columns if "_pred" in c]].apply(lambda x: [float(p) for p in x], axis=1).tolist()
    labels = data["labels"].tolist()
    metrics["macro_f1"] = f1_score(labels, preds, average="macro")

    return metrics


def plot_variable_distribution(df, column_name, title, xlabel, ylabel):
    """Plots the distribution of a specific variable in a DataFrame using plotnine.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name for which the distribution is to be plotted.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    # Calculate the percentage of each value
    value_counts = df[column_name].value_counts().reset_index()
    value_counts.columns = [column_name, "count"]
    value_counts["percentage"] = (value_counts["count"] / value_counts["count"].sum()) * 100

    # Create the bar plot
    plot = (
        ggplot(value_counts, aes(x=column_name, y="count", fill=column_name))
        + geom_bar(stat="identity", show_legend=False)
        + geom_text(aes(label=value_counts["percentage"].apply(lambda x: f"{x:.1f}%")), va="bottom", size=10)
        + labs(title=title, x=xlabel, y=ylabel)
        + theme_light()
    )
    return plot
