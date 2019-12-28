import pathlib
from typing import List, Union, AnyStr

import pandas as pd
from typing.io import IO


def parse_input(s) -> List[tuple]:
    """
    Parses the search input query and creates a data structure:
    ( column_to_search_in, string_to_search, should we filter filter for quoted string)
    :param s: input string to parse
    :return: the list of the search terms to perform and where
    """

    # implementation for 'AND'
    combined_queries = s.split(' AND ')

    queries_to_perform = []

    # find content operators (e.g. "title:")
    import re
    regex = r"([a-z]+):([a-zA-Z0-9 _]+( |$))"

    for query in combined_queries:

        matches = re.finditer(regex, query, re.MULTILINE)

        for match in matches:
            query = list(match.groups())

            # match 0 is the column
            # match 1 is the string to query
            queries_to_perform.append((query[0], query[1], False))

    # assumption: quoted queries are not combined with search operators
    if not queries_to_perform:
        if s.startswith('"') and s.endswith('"'):
            s.replace('"', '')  # remove quotes
            queries_to_perform.append(('content', s, True))
        else:
            queries_to_perform.append(('content', s, False))

    return queries_to_perform


def ranker(df: pd.DataFrame, models: dict, search_query: List[tuple], nr_results=20) -> pd.DataFrame:
    sim_results = []
    cols_to_return = ['rank', 'id', 'author', 'publication', 'title', 'content']

    for col, term, _ in search_query:
        print(f"Searching for '{term}' in '{col}'...")

        # try catch here
        sims = models[col].query_similarity(term)
        sim_results.append(sims)

    # simplest combination of results is the sum of queries
    sim_results = sum(sim_results)

    df['rank'] = sim_results

    # filter for results with similarities
    df = df[df['rank'] > 0]

    # assumption: quoted queries are not combined with search operators
    # is it a quoted query?
    _, term, quoted = search_query.pop()

    if quoted:
        # then we filter for contents that have exactly that string
        term = term.replace('"', '')
        mask = df['content'].apply(lambda x: term in x)
        df = df[mask]

    print(f'Found {len(df)} documents')  # as requested
    return df[cols_to_return].sort_values(by=['rank'], ascending=False).head(nr_results)


def all_news_filereader(filename: Union[str, pathlib.Path, IO[AnyStr]]) -> pd.DataFrame:
    """
    Reads a csv file into a pandas dataframe
    :param filename: path to the filename
    :return: a pandas dataframe
    """
    import pandas as pd

    print(f'Reading file {filename}...')

    df = pd.read_csv(filename)
    df.reset_index(drop=True, inplace=True)

    return df


def read_input() -> str:
    """
    Reads input from CLI
    :return: str of the read input
    """
    print('search: ', end="")
    return input()
