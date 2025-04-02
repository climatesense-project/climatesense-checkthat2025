from typing import Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from plotnine import aes, geom_bar, geom_text, ggplot, labs, theme_light
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def plot_variable_distribution(df, column_name, title, xlabel, ylabel) -> ggplot:
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


def create_multilabel_folds(dataset: Dataset, n_splits: int = 5, random_state: int = None) -> List[DatasetDict]:
    """Create stratified folds for the given dataset.

    Args:
        dataset (Dataset): The dataset to split into folds.
        n_splits (int): Number of folds. Default is 5.
        random_state (int): Random state for reproducibility.

    Returns:
        list: A list of DatasetDict objects, each containing train and test splits.
    """
    folds = MultilabelStratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    labels = dataset["labels"]
    splits = folds.split(range(len(dataset)), labels)

    folds_ds = []
    for _, (train_index, test_index) in enumerate(splits):
        fold_ds = DatasetDict({"train": dataset.select(train_index), "test": dataset.select(test_index)})
        folds_ds.append(fold_ds)
    return folds_ds


def compute_metrics(y_pred, y_test, labels: List[str] = None) -> Dict[str, float]:
    """Compute evaluation metrics for multi-label classification.

    Args:
        y_pred (np.ndarray): Predicted labels, shape (n_samples, n_labels).
        y_test (np.ndarray): True labels, shape (n_samples, n_labels).
        labels (List[str], optional): List of label names. If None, numeric indices are used.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score for each label,
        as well as the macro accuracy, precision, recall, and F1 score across all labels.
    """
    y_pred = np.array(y_pred, copy=None)
    y_test = np.array(y_test, copy=None)
    metrics = {}

    if (labels is None) or (len(labels) != y_test.shape[1]):
        labels = list(range(0, y_test.shape[1]))

    for i in range(0, y_test.shape[1]):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        prec = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)

        metrics.update(
            {
                f"{labels[i]}_avg_acc": acc,
                f"{labels[i]}_avg_prec": prec,
                f"{labels[i]}_avg_rec": rec,
                f"{labels[i]}_avg_f1": f1,
            }
        )
    metrics["macro_acc"] = accuracy_score(y_test, y_pred)
    metrics["macro_prec"] = precision_score(y_test, y_pred, average="macro")
    metrics["macro_rec"] = recall_score(y_test, y_pred, average="macro")
    metrics["macro_f1"] = f1_score(y_test, y_pred, average="macro")

    return metrics
