# Task 1: Subjectivity in News Articles

Data is in the main repository of the task: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/ca6c69e07e2331f59772c0f1cfe7c2b8bf12a8a1/task1/data

Systems are challenged to distinguish whether a sentence from a news article expresses the subjective view of the author behind it or presents an objective view on the covered topic instead.

This is a binary classification tasks in which systems have to identify whether a text sequence (a sentence or a paragraph) is subjective (**SUBJ**) or objective (**OBJ**).

The task comprises three settings:
- **Monolingual**: train and test on data in a given language L
- **Multilingual**: train and test on data comprising several languages
- **Zero-shot**: train on several languages and test on unseen languages

# How to run

- `baseline/baseline.py`: trains the baseline model on input train data and computes prediction on input test data.

- `scorer/evaluate.py`: runs data format checkers and compute metrics based on given ground-truth and predictions.

## Citation

If you use this software, please cite ([bib file](./burel2025clef.bib)):

    Grégoire Burel, Pasquale Lisena, Enrico Daga, Raphael Troncy, Harith Alani. 
    ClimateSense at CheckThat! 2025: Combining Fine-tuned Large Language Models and Conventional Machine Learning Models for Subjectivity and Scientific Web - Discourse Analysis.
    In: CLEF 2025 Working Notes, Ceur-WS Sep 2025, Madrid, Spain.

## Credits

Please find it on the task website: https://checkthat.gitlab.io/clef2025/task1/
