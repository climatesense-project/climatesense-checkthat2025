import re
from functools import partial

import en_core_web_sm
import pandas as pd
from nltk.tokenize import sent_tokenize

nlp = en_core_web_sm.load()

exclude_methods = [
    "life story",
    "sample size",
    "ideal type",
    "life experiences",
    "data privacy",
    "role playing",
    "data quality",
    "survey results",
    "research grants",
    "false negative",
]


def load_methods(file_path="sc_methods.txt"):
    with open(file_path) as f:
        methods = f.read().splitlines()

    return [method for method in methods if method not in exclude_methods]


# methods = load_methods()


def mentions_scientist(tweet_sentence):
    scientists = [
        "research team",
        "research group",
        "scientist",
        "researcher",
        "psychologist",
        "chemist",
        "physician",
        "biologist",
        "economist",
        "engineer",
        "physicist",
        "geologist",
    ]
    for term in scientists:
        res = re.search(r"\s(" + term + r")s?\s", tweet_sentence)
        if res is not None:
            doc = nlp(tweet_sentence)
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    if token.text == term or token.text == term.split(" ")[0]:
                        return term
    return ""


def mentions_science_research_in_general(tweet_sentence):
    science_research_in_general = [
        "research on",
        "research in",
        "research for",
        "research from",
        "research of",
        "research to",
        "research at",
        "research by",
        "research",
        "science of",
        "science to",
        "science",
        "sciences of",
        "sciences to",
        "sciences",
    ]
    for term in science_research_in_general:
        res = re.search(r"\s(" + term + r")s?\s[a-zA-Z]*", tweet_sentence)
        if res is not None:
            doc = nlp(tweet_sentence)
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    if token.text == term or token.text == term.split(" ")[0]:
                        return term
    return ""


def mentions_research_method(tweet_sentence, methods):
    for term in methods:
        if " " in term:
            if " " + term + " " in tweet_sentence:
                return term

    return ""


def mentions_publications(tweet_sentence):
    science_research_in_general = [
        "publications",
        "posters",
        "reports",
        "statistics",
        "datasets",
        "findings",
        "papers",
        "studies",
        "experiments",
        "surveys",
    ]
    for term in science_research_in_general:
        res = re.search(r"\s(" + term + r")s?\s[a-zA-Z]*", tweet_sentence)
        if res is not None:
            doc = nlp(tweet_sentence)
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    if token.text == term or token.text == term.split(" ")[0]:
                        return term
    return ""


def is_related_to_research(tweet, methods):
    tweet = tweet.lower()
    sentences = sent_tokenize(tweet)
    for sent in sentences:
        # md = mentions_discovery(sent)
        # if md != "":
        #     res = True

        msrig = mentions_science_research_in_general(sent)
        if msrig != "":
            msrig = True
        else:
            msrig = False

        ms = mentions_scientist(sent)
        if ms != "":
            ms = True
        else:
            ms = False

        mp = mentions_publications(sent)
        if mp != "":
            mp = True
        else:
            mp = False

        mrm = mentions_research_method(sent, methods=methods)
        if mrm != "":
            mrm = True
        else:
            mrm = False

    return (msrig or ms or mp or mrm), msrig, ms, mp, mrm


def annotate_tweets(tweets, methods):
    res = tweets["text"].progress_apply(partial(is_related_to_research, methods=methods))
    res = list(map(list, zip(*res.values)))
    tweets["is_related_to_research"] = res[0]
    tweets["mentions_science_research_in_general"] = res[1]
    tweets["mentions_scientist"] = res[2]
    tweets["mentions_publications"] = res[3]
    tweets["mentions_research_method"] = res[4]
    return tweets


if __name__ == "__main__":
    # tweets dataframe need a "text" column that contains the text of the tweet
    tweets = pd.read_csv("...", sep="\t")

    print("run heuristics")
    tweets = annotate_tweets(tweets)

    tweets["is_cat3"] = tweets[
        [
            "mentions_discovery",
            "mentions_science_research_in_general",
            "mentions_scientist",
            "mentions_publications",
            "mentions_research_method",
        ]
    ].any(axis="columns")

    cat3_tweets = tweets[tweets["is_cat3"]]
