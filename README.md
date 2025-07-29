# 🪸 ClimateSense CheckThat! 2025

Task 1 consists in distinguishing whether a sentence from a news article expresses the subjective view of the author behind it or presents an objective view on the covered topic instead.

Task 4 consists of two subtasks:

* **Subtask 4a (Scientific Web Discourse Detection):** Given a social media post (tweet), detect if it contains (1) a scientific claim, (2) a reference to a scientific study / publication, or (3) mentions of scientific entities, e.g. a university or scientist.
* **Subtask 4b (Scientific Claim Source Retrieval):** Given an implicit reference to a scientific paper, i.e., a social media post (tweet) that mentions a research publication without a URL, retrieve the mentioned paper from a pool of candidate papers.

Challenge website: https://checkthat.gitlab.io/clef2025/task4/

Challenge GitLab: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab

## ⚙️ Code Style

* Package management with [Poetry](https://python-poetry.org/) (install poetry, then poetry install) then poetry shell to activate the virtual environment).
* Semantic versioning using [Semantic Release](https://python-semantic-release.readthedocs.io/en/latest/).
* Commit messages using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) (make sure your commit messages are formatted correctly).
* Use commitzen to create commit messages (install commitizen, then git cz to create a commit message).

## 📦 Resources

* SciTweets: https://github.com/AI-4-Sci/SciTweets/tree/main
* SciWebClaims: https://github.com/AI-4-Sci/SciWebClaims (can be used for the science content identification task)
* Data for the task 4b is extracted from https://github.com/allenai/cord19.

## 🛠️ Approaches

* Task 1: E5 + MLP classifier.
* Task 4a: Consider SetFit / Look at nvidia/domain-classifier.
* Task 4b: Consider using re-ranker models (see https://huggingface.co/blog/train-reranker).

## Citation

If you use this software, please cite ([bib file](./burel2025clef.bib)):

    Grégoire Burel, Pasquale Lisena, Enrico Daga, Raphael Troncy, Harith Alani. 
    ClimateSense at CheckThat! 2025: Combining Fine-tuned Large Language Models and Conventional Machine Learning Models for Subjectivity and Scientific Web - Discourse Analysis.
    In: CLEF 2025 Working Notes, Ceur-WS Sep 2025, Madrid, Spain.
