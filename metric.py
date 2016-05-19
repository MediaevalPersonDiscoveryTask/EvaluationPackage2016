from __future__ import division  # <== important !
import pandas as pd
import numpy as np
import editdistance


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return start


class AveragePrecision(object):
    """

    Parameters
    ----------
    reference : str
        Path to reference.
    K : iterable, optional
        Compute average precision at rank k for all value k in iterable.
        Defaults to [1, 10, 100].

    Usage
    -----
    >>> average_precision = AveragePrecision('/path/to/reference.txt')
    >>> generator = average_precision('/path/to/hypothesis')
    >>> for query in ['herve_bredin', 'claude_barras', 'camille_guinaudeau']:
            values, n_relevant = generator.send(query)
    """
    def __init__(self, reference, K=[1, 10, 100]):
        super(AveragePrecision, self).__init__()
        self.reference = reference
        self.K = list(K)

        self.reference_ = self._load_reference(reference).groupby(['person_name'])

    def _load_reference(self, path):
        names = ['corpus_id', 'video_id', 'shot_id', 'person_name']
        reference = pd.read_table(path, delim_whitespace=True,
                                  header=None, names=names,
                                  dtype=str)
        return reference

    def _load_hypothesis(self, path):
        # load
        names = ['corpus_id', 'video_id', 'shot_id', 'person_name', 'confidence']
        dtype={'corpus_id': str,
               'video_id': str,
               'shot_id': str,
               'person_name': str,
               'confidence': np.float32}
        hypothesis = pd.read_table(path, delim_whitespace=True,
                                   header=None, names=names,
                                   dtype=dtype)

        # pre-compute length of person name
        # (will make future normalized edit distance computation faster)
        hypothesis['__length'] = hypothesis['person_name'].apply(len)

        # pre-compute shot rank for (person_name, confidence, video_id) ties
        hypothesis['__rank'] = 0
        def __precompute_rank(group):
            if group.shape[0] > 1:
                group['__rank'] = group['shot_id'].argsort()
            return group
        by = ['person_name', 'confidence', 'video_id']
        hypothesis = hypothesis.groupby(by=by).apply(__precompute_rank)

        return hypothesis

    @staticmethod
    def _distance_to(query):
        query_length = len(query)
        def func(hypothesis):
            distance = editdistance.eval(query, hypothesis['person_name'])
            return distance / max(query_length, hypothesis['__length'])
        return func

    def __average_precision(self, query, hypothesis, K=[1, 10, 100]):

        try:
            # get all relevant shots for this query
            relevant = self.reference_.get_group(query)
        except KeyError as e:
            # in case there is no relevant shot,
            # returns perfect average precision
            return [1.0 for _ in K], 0

        # number of relevant/returned shots
        n_relevant = relevant.shape[0]
        n_returned = hypothesis.shape[0]

        # compute distance to query
        hypothesis['__distance'] = hypothesis.apply(self._distance_to(query), axis=1)

        # compute relevance to query
        hypothesis['__relevance'] = False
        for _, r in relevant.iterrows():
            hypothesis.loc[
                (hypothesis['corpus_id'] == r['corpus_id']) &
                (hypothesis['video_id'] == r['video_id']) &
                (hypothesis['shot_id'] == r['shot_id']),
                '__relevance'] = True

        # sort by edit distance first, then confidence, then rank,
        # then (arbitrarily but deterministically) vidoe_id
        hypothesis = hypothesis.sort_values(
            by=['__distance', 'confidence', '__rank', 'video_id'],
            ascending=[True, False, True, True],
            axis=0)

        # compute average precision at various K
        average_precision = []
        for k in K:

            # make sure we do not look further than the number of returned
            _k = min(n_returned, k)

            # actual computation
            relevance =  np.array(hypothesis[:_k]['__relevance'])
            value = np.sum(relevance * relevance.cumsum() / (np.arange(_k) + 1)) / min(n_relevant, _k)

            # store average precision at current K
            average_precision.append(value)

        return average_precision, n_relevant

    @coroutine
    def __call__(self, path):
        """
        Parameters
        ----------
        path : str
            Path to hypothesis file.
        """
        hypothesis = self._load_hypothesis(path)
        query = yield
        while True:
            average_precision, n_relevant = self.__average_precision(query, hypothesis, K=self.K)
            query = yield average_precision, n_relevant
            if query is None:
                break
