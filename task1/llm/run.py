import argparse
import csv
import logging
import random
from pathlib import Path
from typing import AnyStr
from tqdm import tqdm
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

random.seed(0)


def sanitize_and_check_filepath(
        filepath: AnyStr
) -> Path:
    filepath = Path(filepath).resolve()

    if not filepath.exists() or not filepath.is_file():
        raise RuntimeError(f'Could not find file. Got {filepath}')

    return filepath


def run(data_dir: Path, train_filepath: Path, test_filepath: Path, language='english', llm_model='mistral') -> Path:
    train_data = pd.read_csv(train_filepath, sep='\t', quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv(test_filepath, sep='\t', quoting=csv.QUOTE_NONE)

    obj_example = 'For half a century, household expenditure on gas and electricity has, in real terms, hovered around the £1,000 mark.'
    subj_example = 'But the party must do more than talk.'

    llm = OllamaLLM(model=llm_model)
    prompt_template = ("Distinguish whether a sentence from a news article expresses the subjective view (SUBJ) of the author behind it "
                       "or presents an objective view (OBJ) on the covered topic instead."
                       "Here some examples:\n"
                       "<<" + obj_example + ">> is an OBJ sentence"
                       "<<" + subj_example + ">> is an SUBJ sentence"
                       "The sentence to analyse is in "+language+": {sentence}."
                       "Strictly return OBJ or SUBJ and nothing else. I do not need any explanation."
                       "It is important to not return anything more than the string 'SUBJ' or 'OBJ'.")
    prompt = PromptTemplate(
        input_variables=["sentence"], template=prompt_template
    )
    chain = prompt | llm | StrOutputParser()

    preds = []
    limit = 10 # the max is 600
    for x in tqdm(test_data.head(limit)['sentence']):
        returned = chain.invoke(x)
        preds.append((returned[0:4].strip('.').strip('E').strip(),returned))

    predictions_filepath = data_dir.joinpath(f'{llm_model}_pred.tsv')
    with open(predictions_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(preds)

    for p,l, sent in zip(preds, test_data.head(limit)['label'], test_data.head(limit)['sentence']):
        print(f"{p[0]} {l} {p==l} {sent}")

    acc = accuracy_score(test_data.head(limit)['label'], [p[0] for p in preds])

    print(f'Accuracy: {acc:.4f}')
    # model.fit(X=vect.encode(train_data['sentence'].values), y=train_data['label'].values)
    #
    # predictions = model.predict(X=vect.encode(test_data['sentence'].values)).tolist()
    # pred_df = pd.DataFrame()
    # pred_df['sentence_id'] = test_data['sentence_id']
    # pred_df['label'] = predictions
    #
    # predictions_filepath = data_dir.joinpath('baseline_pred.tsv')
    # pred_df.to_csv(predictions_filepath, index=False, sep='\t', quoting=csv.QUOTE_NONE)

    # return predictions_filepath


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', '-trp',
                        required=True,
                        type=str, )
    parser.add_argument('--testpath', '-ttp',
                        required=True,
                        type=str, )
    parser.add_argument('--language', '-lang',
                        default='english',
                        type=str, )
    parser.add_argument('--llm', '-llm',
                        default='mistral',
                        type=str, )
    args = parser.parse_args()

    train_filepath = sanitize_and_check_filepath(args.trainpath)
    test_filepath = sanitize_and_check_filepath(args.testpath)
    lang = args.language
    llm = args.llm

    data_dir = test_filepath.parent

    logging.info(f"""Running baseline with following configuration: 
                 Train: {train_filepath} 
                 Test: {test_filepath}""")
    run(data_dir=data_dir,
        test_filepath=test_filepath,
        train_filepath=train_filepath,
        language=lang,
        llm_model=llm
        )
