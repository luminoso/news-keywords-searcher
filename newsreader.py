from typing import Dict

import pandas as pd

from auxfunctions import all_news_filereader, read_input
from auxfunctions import parse_input
from auxfunctions import ranker
from td import Model
from td import Tokenizer


def build_models(df: pd.DataFrame, usecols: list) -> Dict:
    """
    For each column build a model.
    Adds each model to a dictionary
    :param df: original dataframe with all data
    :param usecols: columns to build a model
    :return: dictionary with models for each column
    """
    models = {}
    tokenizer = Tokenizer()

    for col in usecols:
        print(f'Tokenizing and building model for {col}...')

        model = Model(tokenizer)

        corpus = df[col].astype('unicode').apply(lambda x: model.build_indexes(x))
        model.build_similarity(list(corpus), model='tfidf')

        models[col] = model

    return models


def main(filename):
    # read file
    df = all_news_filereader(filename)

    # df = df[0:1000]

    models = build_models(df, usecols=['title', 'publication', 'author', 'content'])

    # prettier print
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    while True:
        s = read_input()
        parsed_query = parse_input(s)
        results = ranker(df, models, parsed_query)
        print(results)
        print()


if __name__ == "__main__":
    try:
        import sys

        if len(sys.argv) != 2:
            print('Usage: python newsreader.py <filename>')
            exit(1)

        # collect filename to read from arguments
        main(sys.argv[1])

    except KeyboardInterrupt:
        print('exiting')
