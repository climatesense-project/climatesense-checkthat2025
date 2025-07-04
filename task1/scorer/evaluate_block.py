import os
from evaluate import evaluate, validate_files

langmap = {
  # "bulgarian": ["bg", "zero_"],
  "german": ["de", "mono_"],
  "romanian": ["ro", "zero_"],
  "multilingual": ["multilingual", ""],
  "italian": ["it", "mono_"],
  "english": ["en", "mono_"],
  "greek": ["gr", "zero_"],
  "polish": ["pol", "zero_"],
  "arabic": ["ar", "mono_"],
  "ukrainian": ["ukr", "zero_"]
}


for lang in os.listdir("data"):
    if lang not in langmap:
        continue
    mapped =langmap[lang]
    source = f'data/{lang}/test_{mapped[0]}_labeled.tsv'
    target = f'submission/task1_test_{mapped[1]}{lang}.tsv'

    whole_data = validate_files(target, source)
    print(len(whole_data))
    whole_data = whole_data[whole_data["sentence"].str.len()<40]
    print(len(whole_data))

    res = evaluate(whole_data)
    print(lang, '\t', res['macro-F1'])