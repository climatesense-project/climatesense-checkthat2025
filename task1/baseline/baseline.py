import argparse
import csv
import logging
import random
from pathlib import Path
from typing import AnyStr
import pickle
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import dl_translate as dlt
from deep_translator import GoogleTranslator as Translate


random.seed(0)

mt = dlt.TranslationModel()  # Slow when you load it for the first time

def sanitize_and_check_filepath(
        filepath: AnyStr
) -> Path:
    filepath = Path(filepath).resolve()

    if not filepath.exists() or not filepath.is_file():
        raise RuntimeError(f'Could not find file. Got {filepath}')

    return filepath


def run_sbert_lr_baseline(
        data_dir: Path,
        train_filepath: Path,
        test_filepath: Path,
        name: AnyStr,
        language: AnyStr
) -> Path:
    train_data = pd.read_csv(train_filepath, sep='\t', quoting=csv.QUOTE_NONE)
    test_data = pd.read_csv(test_filepath, sep='\t', quoting=csv.QUOTE_NONE)

    vect = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    # vect = SentenceTransformer("digitalepidemiologylab/covid-twitter-bert")

    model = MLPClassifier(random_state=1, max_iter=100, solver='adam')
    # model = SVC()
    # model = LogisticRegression(class_weight="balanced")
    model.fit(X=vect.encode(train_data['sentence'].values), y=train_data['label'].values)

    model_filepath = data_dir.joinpath(f'{name}_model.pkl')
    pickle.dump(model, open(model_filepath, 'wb'))

    test_sentences = test_data['sentence'].values
    if language and language != 'en':
        print('Translating')
        test_sentences = [Translate(source=language, target='en').translate(sent) for sent in tqdm(test_sentences)]
        with open(data_dir.joinpath(f'translated.txt'), 'w') as f:
            for x in test_sentences:
                f.write(x)
                f.write('\n')

    predictions = model.predict(X=vect.encode(test_sentences)).tolist()
    pred_df = pd.DataFrame()
    pred_df['sentence_id'] = test_data['sentence_id']
    pred_df['label'] = predictions

    predictions_filepath = data_dir.joinpath(f'{name}_pred.tsv')
    pred_df.to_csv(predictions_filepath, index=False, sep='\t', quoting=csv.QUOTE_NONE)

    return predictions_filepath


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', '-trp',
                        required=True,
                        type=str, )
    parser.add_argument('--testpath', '-ttp',
                        required=True,
                        type=str, )
    parser.add_argument('--name', '-n',
                        required=True,
                        type=str, )
    parser.add_argument('--language', '-l',
                        type=str, )
    args = parser.parse_args()

    train_filepath = sanitize_and_check_filepath(args.trainpath)
    test_filepath = sanitize_and_check_filepath(args.testpath)

    data_dir = test_filepath.parent

    logging.info(f"""Running baseline with following configuration: 
                 Train: {train_filepath} 
                 Test: {test_filepath}""")
    run_sbert_lr_baseline(data_dir=data_dir,
                          test_filepath=test_filepath,
                          train_filepath=train_filepath,
                          name=args.name,
                          language=args.language)
