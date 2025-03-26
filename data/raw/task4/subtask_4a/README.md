# Subtask 4a (Scientific Web Discourse Detection)

Given a social media post (tweet), detect if it contains (1) a scientific claim, (2) a reference to a scientific study / publication, or (3) mentions of scientific entities, e.g. a university or scientist.

The task is a multilabel classification task.

__Table of contents:__

<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Task 4a Directory](#contents-of-the-repository)
- [Datasets statistics](#datasets-statistics)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)

<!-- ## Evaluation Results

TBA -->

## List of Versions
- [20/01/2025] Training data released.

## Contents of the Subtask4a Directory

This repository contains the following files:

1. **baselines.ipynb** Jupyter notebook that enables participants of subtask 4a to quickly get started
2. **ct_train_data.tsv** The training data (tweet texts + ground-truth labels)
3. **README.md** this file

## Datasets statistics
The training data **ct_train_data.tsv** contains 1,366 tweets + ground-truth labels.

## Input Data Format

The tweets are provided as a .tsv file with three columns
> index <TAB> text <TAB> labels

Where: <br>
* index: index for the data samples (used for internal purposes)
* text: the preprocessed tweet text
* labels: a list of three labels, one for each category (e.g., [0.0, 0.0, 1.0])

**Example:**
See examples at: https://checkthat.gitlab.io/clef2025/task4/#subtask-4a-dataset-examples

(user mentions are anonymized with @user, tweets that are quotes contain the word "quote" at the end of the text, tweets that contain image(s) contain the word "image" at the end of the text)
## Output Data Format

The output must be a TSV format with two columns: index and predictions. The values in the predictions column should be formatted as in the original labels column (e.g., [0.0, 0.0, 1.0]).

## Evaluation Metrics

This task is a multilabel classification task and will be evaluated using the macro-averaged F1-score.

## Baselines & Scorers

The **baselines.ipynb** jupyter notebook can be found in this repository. Participants can download it and use it to get started quickly. The notebook contains:

1. Code to load the data from ct_train_data.tsv
2. Code to train and evaluate various models (models available on Huggingface) using 5-fold crossvalidation.


## Submission

TBA

## Related Work

Information regarding the three categories can be found in the original [publication](https://dl.acm.org/doi/10.1145/3511808.3557693)

## Contact
Please contact sebastian.schellhammer@gesis.org

## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task4/
