# Task 1: Subjectivity in News Articles

Data is in the main repository of the task: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/ca6c69e07e2331f59772c0f1cfe7c2b8bf12a8a1/task1/data


Systems are challenged to distinguish whether a sentence from a news article expresses the subjective view of the author behind it or presents an objective view on the covered topic instead.

This is a binary classification tasks in which systems have to identify whether a text sequence (a sentence or a paragraph) is subjective (**SUBJ**) or objective (**OBJ**).

The task comprises three settings:
- **Monolingual**: train and test on data in a given language L
- **Multilingual**: train and test on data comprising several languages
- **Zero-shot**: train on several languages and test on unseen languages

__Table of contents:__

<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Task 1 Directory](#contents-of-the-repository)
- [Datasets statistics](#datasets-statistics)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)


## List of Versions
- [04/02/2025] Arabic dataset updated (We used the dataset released here: https://arxiv.org/pdf/2406.05559).
- [13/01/2025] Training data released.



## Contents of the Task 1 Directory

- Main folder: [data](./data)
  - Contains a subfolder for each language which contain the data as TSV format with .tsv extension (train_LANG.tsv, dev_LANG.tsv, dev_test_LANG.tsv, test_LANG.tsv).
  As LANG we used standard language code for each language.
- Main folder: [baseline](./baseline)<br/>
  - Contains a single file, baseline.py, used to train a baseline and provide predictions.
- Main folder: [scorer](./scorer)<br/>
  - Contains a single file, evaluate.py, that checks the format of a submission and evaluate the various metrics.
- [README.md](./README.md) <br/>

## Datasets statistics

* **English**
  - train: 830 sentences, 532 OBJ, 298 SUBJ
  - dev: 462 sentences, 222 OBJ, 240 SUBJ
  - dev-test: 484 sentences, 362 OBJ, 122 SUBJ
  - test: TBA
* **Italian**
  - train: 1613 sentences, 1231 OBJ, 382 SUBJ
  - dev: 667 sentences, 490 OBJ, 177 SUBJ
  - dev-test - 513 sentences, 377 OBJ, 136 SUBJ
  - test: TBA
* **German**
  - train: 800 sentences, 492 OBJ, 308 SUBJ
  - dev: 491 sentences, 317 OBJ, 174 SUBJ
  - dev-test - 337 sentences, 226 OBJ, 111 SUBJ
  - test: TBA
* **Bulgarian**
  - train: 729 sentences, 406 OBJ, 323 SUBJ
  - dev: 467 sentences, 175 OBJ, 139 SUBJ
  - dev-test - 250 sentences, 143 OBJ, 107 SUBJ
  - test: TBA
* **Arabic**
  - train: 2,446 sentences, 1391 OBJ, 1055 SUBJ
  - dev: 742 sentences, 266 OBJ, 201 SUBJ
  - dev-test - 748 sentences, 425 OBJ, 323 SUBJ
  - test: TBA

## Input Data Format

The data will be provided as a TSV file with three columns:
> sentence_id <TAB> sentence <TAB> label

Where: <br>
* sentence_id: sentence id for a given sentence in a news article<br/>
* sentence: sentence's text <br/>
* label: *OBJ* and *SUBJ*

<!-- **Note:** For English, the training and development (validation) sets will also include a fourth column, "solved_conflict", whose boolean value reflects whether the annotators had a strong disagreement. -->

**Examples:**

> b9e1635a-72aa-467f-86d6-f56ef09f62c3  Gone are the days when they led the world in recession-busting SUBJ
>
> f99b5143-70d2-494a-a2f5-c68f10d09d0a  The trend is expected to reverse as soon as next month.  OBJ

## Output Data Format

The output must be a TSV format with two columns: sentence_id and label.

## Evaluation Metrics

This task is evaluated as a classification task. We will use the F1-macro measure for the ranking of teams.

We will also measure Precision, Recall, and F1 of the SUBJ class and the macro-averaged scores.
<!--
There is a limit of 5 runs (total and not per day), and only one person from a team is allowed to submit runs.

Submission Link: Coming Soon

Evaluation File task3/evaluation/CLEF_-_CheckThat__Task3ab_-_Evaluation.txt -->

## Scorers

To evaluate the output of your model which should be in the output format required, please run the script below:

> python evaluate.py -g dev_truth.tsv -p dev_predicted.tsv

where dev_predicted.tsv is the output of your model on the dev set, and dev_truth.tsv is the golden label file provided by us.

The file can be used also to validate the format of the submission, simply use the provided test file as gold data.
The evaluation will not be performed, but the format of your input will be checked.


## Baselines

The script to train the baseline is provided in the related directory.
The script can be run as follow:

> python baseline.py -trp train_data.tsv -ttp dev_data.tsv

where train_data.tsv is the file to be used for training and dev_data.tsv is the file on which doing the prediction.

The baseline is a logistic regressor trained on a Sentence-BERT multilingual representation of the data.

<!-- ### Task 3: Multi-Class Fake News Detection of News Articles

For this task, we have created a baseline system. The baseline system can be found at https://zenodo.org/record/6362498
 -->

## Submission

### Scorers, Format Checkers, and Baseline Scripts

- ``baseline/baseline.py``: trains the baseline model on input train data and computes prediction on input test data.
- ``scorer/evaluate.py``: runs data format checkers and compute metrics based on given ground-truth and predictions.

### Submission Guidelines

- Make sure that you create one account for each team, and submit it through one account only.
- The last file submitted to the leaderboard will be considered as the final submission.
- Name of the output file has to be `task1_{SETTING}.tsv` with `.tsv` extension (e.g., ``task1_arabic.tsv``); otherwise, you will get an error on the leaderboard.
In particular, the settings are: ``mono_arabic``, ``mono_english``, ``mono_german``, ``mono_italian``, ``multilingual``, ``zero_greek``, ``zero_polish``, ``zero_ukrainian``, and ``zero_romanian``.
- You have to zip the tsv (e.g., `zip task1_mono_arabic.zip`), zip and submit it through the codalab page.
- It is required to submit the team name for each submission and fill out the [questionnaire](https://forms.gle/tbC4dirDsuCHQsab8) to provide some details on your approach as we need that information for the overview paper.
- You are allowed to submit max 200 submissions per day for each subtask.
- We will keep the leaderboard private till the end of the submission period, hence, results will not be available upon submission. All results will be available after the evaluation period.

### Submission Site
The submission is done through the Codalab platform at https://codalab.lisn.upsaclay.fr/competitions/22756

## Leaderboard

We report macro-F1 score to compare models on each setting.

### Dev-test

| **Model**     | **Arabic** | **Bulgarian** | **English** | **German** | **Italian** |
|---------------|------------|---------------|-------------|------------|-------------|
| MiniLM-L12-v2 | 0.55       | 0.75          | 0.63        | 0.69       | 0.63        |

### Monolingual

ITALIAN
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.8104                 |
| aelboua                | CEA-LIST              | 0.8075                 |
| smollab                | smollab               | 0.7750                 |
| tomasbernal01          | UmuTeam               | 0.7703                 |
| Ather-Hashmi           | Investigators         | 0.7468                 |
| Arcturus               | Arcturus              | 0.7282                 |
| msmadi                 | msmadi                | 0.7139                 |
| matteofasulo           | AI Wizards            | 0.7130                 |
| cepanca                | UNAM                  | 0.7086                 |
| srijani                | JU_NLP                | 0.6991                 |
|                        | Baseline              | 0.6941                 |
| rtroncy                | ClimateSense          | 0.6839                 |
| Bharatdeep_Hazarika    | TIFIN INDIA           | 0.5808                 |
| Sumitjais              | IIIT Surat            | 0.4612                 |

ARABIC
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| aelboua                | CEA-LIST              | 0.6884                 |
| tomasbernal01          | UmuTeam               | 0.5903                 |
| Ather-Hashmi           | Investigators         | 0.5880                 |
| msmadi                 | msmadi                | 0.5771                 |
| matteofasulo           | AI Wizards            | 0.5646                 |
| Sumitjais              | IIIT Surat            | 0.5456                 |
| Arcturus               | Arcturus              | 0.5376                 |
|                        | Baseline              | 0.5133                 |
| rtroncy                | ClimateSense          | 0.5120                 |
| smollab                | smollab               | 0.5053                 |
| hazemAbdelsalam        | hazemAbdelsalam       | 0.5038                 |
| Bharatdeep_Hazarika    | TIFIN INDIA           | 0.4427                 |
| kishan_g               | kishan_g              | 0.4427                 |
| srijani                | JU_NLP                | 0.4328                 |

GERMAN
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| smollab                | smollab               | 0.8520                 |
| cepanca                | UNAM                  | 0.8280                 |
| msmadi                 | msmadi                | 0.8013                 |
| aelboua                | CEA-LIST              | 0.7733                 |
| matteofasulo           | AI Wizards            | 0.7718                 |
| Ather-Hashmi           | Investigators         | 0.7583                 |
| Bharatdeep_Hazarika    | TIFIN INDIA           | 0.7375                 |
| kishan_q               | kishan_g              | 0.7375                 |
| srijani                | JU_NLP                | 0.7356                 |
| tomasbernal01          | UmuTeam               | 0.7324                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.7269                 |
| rtroncy                | ClimateSense          | 0.7213                 |
| Arcturus               | Arcturus              | 0.7115                 |
| duckLingua             | duckLingua            | 0.7114                 |
|                        | Baseline              | 0.6960                 |
| Sumitjais              | IIIT Surat            | 0.6342                 |

ENGLISH
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| msmadi                 | msmadi                | 0.8052                 |
| kishan_q               | kishan_g              | 0.7955                 |
| aelboua                | CEA-LIST              | 0.7739                 |
| tomasbernal01          | UmuTeam               | 0.7604                 |
| Ather-Hashmi           | Investigators         | 0.7544                 |
| Arcturus               | Arcturus              | 0.7522                 |
| selmey                 | nlu@utn               | 0.7486                 |
| srijani                | JU_NLP                | 0.7334                 |
| smollab                | smollab               | 0.7328                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.7228                 |
| rtroncy                | ClimateSense          | 0.7226                 |
| anpiz                  | NLP-UTB               | 0.7130                 |
| chepanca               | UNAM                  | 0.7075                 |
| R_Padmashri            | CheckMates            | 0.7009                 |
| DSGT-CheckThat         | DSGT-CheckThat        | 0.6830                 |
| Tanvir_77              | CUET_KCRL             | 0.6783                 |
| KnowThySelf            | CSECU-Learners        | 0.6777                 |
| NapierNLP              | NapierNLP             | 0.6724                 |
| matteofasulo           | AI Wizards            | 0.6600                 |
| Sumitjais              | IIIT Surat            | 0.6492                 |
| Bharatdeep_Hazarika    | TIFIN India           | 0.5756                 |
| mariuxi                | UGPLN                 | 0.5531                 |
|                        | Baseline              | 0.5370                 |

### Multilingual

MULTILINGUAL
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| Bharatdeep_Hazarika    | TIFIN India           | 0.7550                 |
| kishan_g               | kishan_g              | 0.7550                 |
| aelboua                | CEA-LIST              | 0.7396                 |
| KnowThySelf            | CSECU-Learners        | 0.7321                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.7186                 |
| smollab                | smollab               | 0.7115                 |
| tomasbernal01          | UmuTeam               | 0.7074                 |
| msmadi                 | msmadi                | 0.6692                 |
| BigO_NLP_              | CSECU-Leaners         | 0.6558                 |
| srijani                | JU_NLP                | 0.6536                 |
| Arcturus               | Arcturus              | 0.6484                 |
| rtroncy                | ClimateSense          | 0.6453                 |
|                        | Baseline              | 0.6390                 |
| Ather-Hashmi           | Investigators         | 0.6292                 |
| Sumitjais              | IIIT Surat            | 0.5411                 |
| matteofasulo           | AI Wizards            | 0.2380                 |

### Zero-shot

POLISH
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| aelboua                | CEA-LIST              | 0.6922                 |
| Sumitjais              | IIIT Surat            | 0.6676                 |
| KnowThySelf            | CSECU-Learners        | 0.6558                 |
| matteofasulo           | AI Wizards            | 0.6322                 |
| Arcturus               | Arcturus              | 0.6298                 |
| Ather-Hashmi           | Investigators         | 0.6055                 |
| tomasbernal01          | UmuTeam               | 0.5763                 |
| smollab                | smollab               | 0.5738                 |
|                        | Baseline              | 0.5719                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.5665                 |
| sriani                 | JU_NLP                | 0.5603                 |
| rtroncy                | ClimateSense          | 0.5525                 |
| msmadi                 | msmadi                | 0.5165                 |
| Bharatdeep_Hazarika    | TIFIN INDIA           | 0.3811                 |

UKRAINIAN
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| KnowThySelf            | CSECU-Learners        | 0.6424                 |
| Ather-Hashmi           | Investigators         | 0.6413                 |
| rtroncy                | ClimateSense          | 0.6395                 |
| matteofasulo           | AI Wizards            | 0.6383                 |
|                        | Baseline              | 0.6296                 |
| smollab                | smollab               | 0.6238                 |
| tomasbernal01          | UmuTeam               | 0.6210                 |
| msmadi                 | msmadi                | 0.6168                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.6124                 |
| aelboua                | CEA-LIST              | 0.6061                 |
| sriani                 | JU_NLP                | 0.5802                 |
| Arcturus               | Arcturus              | 0.5553                 |
| Sumitjais              | IIIT Surat            | 0.5125                 |
| Bharatdeep_Hazarika    | TIFIN INDIA           | 0.4731                 |

ROMANIAN
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| msmadi                 | msmadi                | 0.8126                 |
| KnowThySelf            | CSECU-Learners        | 0.7992                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.7917                 |
| smollab                | smollab               | 0.7892                 |
| tomasbernal01          | UmuTeam               | 0.7793                 |
| aelboua                | CEA-LIST              | 0.7659                 |
| matteofasulo           | AI Wizards            | 0.7507                 |
| srijani                | JU_NLP                | 0.7442                 |
| rtroncy                | ClimateSense          | 0.7396                 |
| Arcturus               | Arcturus              | 0.7366                 |
| Ather-Hashmi           | Investigators         | 0.7133                 |
| Sumitjais              | IIIT Surat            | 0.6496                 |
|                        | Baseline              | 0.6461                 |
| Bharatdeep_Hazarika    | TIFIN INDIA           | 0.5181                 |

GREEK
| **Codalab**            | **Team**              | **F1**                 |
|------------------------|-----------------------|------------------------|
| matteofasulo           | AI Wizards            | 0.5067                 |
| smollab                | smollab               | 0.4945                 |
| KnowThySelf            | CSECU-Learners        | 0.4919                 |
| tomasbernal01          | UmuTeam               | 0.4831                 |
| Doe (Ariana Sahitaj)   | XplaiNLP              | 0.4750                 |
| Ather-Hashmi           | Investigators         | 0.4539                 |
| aelboua                | CEA-LIST              | 0.4492                 |
| srijani                | JU_NLP                | 0.4351                 |
|                        | Baseline              | 0.4159                 |
| rtroncy                | ClimateSense          | 0.4137                 |
| msmadi                 | msmadi                | 0.4057                 |
| Arcturus               | Arcturus              | 0.3905                 |
| Sumitjais              | IIIT Surat            | 0.3733                 |
| Bharatdeep_Hazarika    | TIFIN India           | 0.3337                 |


## Related Work

Information regarding the annotation guidelines can be found in the following papers:

> Federico Ruggeri, Francesco Antici, Andrea Galassi, aikaterini Korre, Arianna Muti, Alberto Barron,  _[On the Definition of Prescriptive Annotation Guidelines for Language-Agnostic Subjectivity Detection](https://ceur-ws.org/Vol-3370/paper10.pdf)_, in: Proceedings of Text2Story — Sixth Workshop on Narrative Extraction From Texts, CEUR-WS.org, 2023, Vol 3370, pp. 103 - 111

> Francesco Antici, Andrea Galassi, Federico Ruggeri, Katerina Korre, Arianna Muti, Alessandra Bardi, Alice Fedotova, Alberto Barrón-Cedeño, _[A Corpus for Sentence-level Subjectivity Detection on English News Articles](https://arxiv.org/abs/2305.18034)_, in: Proceedings of Joint International Conference on Computational Linguistics, Language Resources and Evaluation (COLING-LREC), 2024

> Suwaileh, Reem, Maram Hasanain, Fatema Hubail, Wajdi Zaghouani, and Firoj Alam. "ThatiAR: Subjectivity Detection in Arabic News Sentences." arXiv preprint arXiv:2406.05559 (2024).



## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task1/
