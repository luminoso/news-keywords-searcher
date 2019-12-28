**_news-keywords-searcher_** main objective is a proof of concept project that implements a keyword search for [_All the news - 143,000 articles from 15 American publications_](https://www.kaggle.com/snapcrack/all-the-news/version/4#articles1.csv) Kaggle dataset.

**Basic features**

The implementation receives a search query as input and outputs:
- The total number of news articles in the dataset that match the search query
- The 20 most relevant news in the provided dataset that match the search query
- The ranking score for each one of the _n_ most relevant news articles retrieved

The search interpreter supports 3 kinds of operators: 'title', 'author', and 'publication'. They can be combined with more free text or quoted text. For example:
- "United States of America"
- Trump
- obama AND publication:cnn
- _title:chicago_
- _title:brazil publication:breitbart author:frances_
- _title:death penalty AND content:boston_

 
The implementation leverages on 3 main frameworks:
- [Pandas](https://pandas.pydata.org/): data structure and presenting the results
- [Spacy](https://spacy.io/):  corpus tokenization and lemmatization
- [Gensim](https://radimrehurek.com/gensim/): corpus dictionary, converting articles to bag-of-words (doc2bow) and similarity models

There are two similarity models implemented: [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and [LSI](https://radimrehurek.com/gensim/models/lsimodel.html), both supported by Gensim.

Most obvious limitations:
- There isn't any particular data structure for serialization and manipulation
- No memory restrictions or requirements: the dataset must fit on memory
- Similarities weighting when using search operators are just a sum of multiple ranks
- Single-threaded


# Install and run guide


Install requirements.txt with your favorite python virtual environment tool and run newsreader.py <filename>

```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

# python allnews.py <filename>
python newsreader.py all-the-news/articles1.csv
```

<div style="text-align:center"><img src="https://github.com/luminoso/news-keywords-searcher/blob/master/doc/screenshot.png" /></div>

There's a search interpreter when the program runs directly from CLI

Since the implementation has python type hints and f-strings, the minimum required version is python 3.6

There are examples of programmatically querying the dataset examples in the [_search_examples.ipynb_](https://github.com/luminoso/news-keywords-searcher/blob/master/doc/search_examples.ipynb) Jupyter Notebook. This repo includes a small _'articles1.csv'_ portion of the full Kaggel's dataset with the first 2500 articles. Tokenizing the news articles may take some minutes for the full dataset.


# Small walkthrough though implementation

**1. Reader** ([auxfunctions.py](https://github.com/luminoso/news-keywords-searcher/blob/master/auxfunctions.py))

Based on a pandas dataframe as a reader and as a data structure for the whole PoC

**2. Build models**

One model per corpus. This means that _'title'_, _'publication'_, _'author'_ and _'content'_ share neither models or a dictionary.

_Tokenizer_ and _Model_ classes are implemented in [td.py](https://github.com/luminoso/news-keywords-searcher/blob/master/td.py)

- Tokenizer class
    - From spacy framework
    - Rules shared across different models
    - Text processing loop is: Tokenization -> Lemmatization -> Filter ( e.g. stopwords, punctuation, etc)

- Model class
    - Responsible for applying the tokenization process
    - Incrementally updates dictionary and converts each 'news' text tokens to vectors
    - Builds a similarity model (_tf-idf_, _LSI_ or other) of each corpus

**3. Query similarities and Rank results** ([auxfunctions.py](https://github.com/luminoso/news-keywords-searcher/blob/master/auxfunctions.py))

- The search input is parsed according to some assumptions of the challenge details
    - input can be combined with search operators (e.g.: title:obama) that match the columns of the dataset
    - _'AND'_ keyword splits multiple operators
    - quoted text has the meaning for the search of an exact string
- Implementation is a combination of a splitter and a regex
- Uses a list of tuples as a data structure of the parsed query for querying the models

**4. Ranker** ([auxfunctions.py](https://github.com/luminoso/news-keywords-searcher/blob/master/auxfunctions.py))
- Query similarity between the (parsed) search input and the corpus models
- The quoted text is a query for exact text, implemented via pandas masks
- Don't output if the similarity is 0
- Multiple similarities between the corpus of different columns are weighted by sum

# License
MIT
