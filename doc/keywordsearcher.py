from typing import List

import numpy


class Tokenizer:
    """
    A tokenizer based on Spacy framework
    """

    def __init__(self):
        from spacy.lang.en import English
        nlp = English()

        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        self.tokenizer = nlp.Defaults.create_tokenizer(nlp)

    def __filter_lemma(self, t) -> bool:
        """
        Lemma filter for stopwords, punctuation, etc
        :param t: lemma to test
        :return: false if filtered
        """
        return (t.is_alpha and
                not (t.is_space or t.is_punct or
                     t.is_stop or t.like_num))

    def tokenize(self, s: str, filter_lemma=True, to_lower=True):
        """
        Tokenize a string
        :param s: string to tokenize
        :param filter_lemma: if should we filter for stop works
        :param to_lower: should tokenizer perform lowercase conversion
        :return: list of lemmas for the string
        """

        # convert string to a spacy doc
        doc = self.tokenizer(s)

        # retrieve lemma terms from spacy

        if filter_lemma:
            lemmas = [t.lemma_ for t in doc if self.__filter_lemma(t)]
        else:
            lemmas = [t.lemma_ for t in doc]

        if to_lower:
            lemmas = [t.lower() for t in lemmas]

        return lemmas


class Model:
    def __init__(self, tokenizer):
        from gensim import corpora

        self.tokenizer = tokenizer
        self.dictionary = corpora.Dictionary()
        self.model = None
        self.index = None

    def build_indexes(self, s: str, allow_update=True):
        """
        Converts a string into a bag of words
        :param s: string to convert
        :param allow_update: if dictionary should be updated
        :return: bag of words
        """

        # tokenize and 'lemmanize' the string
        lemmas = self.tokenizer.tokenize(s)

        # return bag of words matching the string
        return self.dictionary.doc2bow(lemmas, allow_update=allow_update)

    def build_similarity(self, corpus: List[tuple], model='tfidf') -> None:
        """
        Builds a similarity model for a bag of words corpus
        :param corpus: to build the similarity model
        :param model: strategy
        """

        from gensim.models.tfidfmodel import TfidfModel
        from gensim.models.lsimodel import LsiModel
        from gensim import similarities

        self.dictionary.compactify()

        if model == 'tfidf':
            self.model = TfidfModel(corpus,
                                    id2word=self.dictionary)
        elif model == 'lsi':
            # todo: remove magic number
            self.model = LsiModel(corpus,
                                  id2word=self.dictionary,
                                  num_topics=2)

        feature_cnt = len(self.dictionary.token2id)
        self.index = similarities.SparseMatrixSimilarity(self.model[corpus],
                                                         num_features=feature_cnt)

    def query_similarity(self, query: str) -> numpy.ndarray:
        """
        Perform a similarity query on me model
        :param query: string to query
        :return: similarities list
        """

        vec_bow = self.build_indexes(query, allow_update=False)
        vec_model = self.model[vec_bow]
        sims = self.index[vec_model]

        return sims
