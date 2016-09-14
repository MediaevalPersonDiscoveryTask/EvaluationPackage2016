#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

""""MediaEval Person Discovery 2016" Evaluation Metric

Usage:
  metric [options] <reference.txt> <hypothesis.txt>
  metric (-h | --help)

Options:

  <reference.txt>           Path to reference file.
  <hypothesis.txt>          Path to hypothesis file.
  --queries=<queries.txt>   Path to list of queries.
  --subset=<videos.txt>     Path to test subset.
  -h --help                 Show this screen.
  --verbose                 Show progress.
"""

from __future__ import division  # <== important !
from __future__ import print_function
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
    subset : iterable of (corpus_id, video_id) tuples
        When provided, evaluation only on this subset
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
    def __init__(self, reference, subset=None, K=[1, 10, 100]):
        super(AveragePrecision, self).__init__()
        self.reference = reference
        self.subset = subset
        self.K = list(K)

        self.reference_ = self._load_reference(reference).groupby(['person_name'])

    @property
    def queries(self):
        return sorted(self.reference_.groups)

    def in_subset(self, row):
        """Return true if current video is part of the evaluated subset"""
        corpus_id, video_id = row['corpus_id'], row['video_id']
        return (corpus_id, video_id) in self.subset

    def _load_reference(self, path):
        names = ['corpus_id', 'video_id', 'shot_id', 'person_name']
        reference = pd.read_table(path, delim_whitespace=True,
                                  header=None, names=names,
                                  dtype=str)

        if self.subset:
            reference = reference[reference.apply(self.in_subset, axis=1)]

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

        if self.subset:
            hypothesis = hypothesis[hypothesis.apply(self.in_subset, axis=1)]

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

            # make sure we do not look further than the number of relevant
            _k = min(n_relevant, k)

            # actual computation
            relevance =  np.array(hypothesis[:_k]['__relevance'])
            value = np.sum(relevance * relevance.cumsum() / (np.arange(_k) + 1)) / _k

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


if __name__ == '__main__':

    from docopt import docopt
    from tqdm import tqdm

    arguments = docopt(__doc__)

    subset = arguments['--subset']
    if subset:
        with open(subset, 'r') as fp:
            subset = [tuple(line.strip().split()) for line in fp]

    reference = arguments['<reference.txt>']
    average_precision = AveragePrecision(reference, subset=subset)

    queries = arguments['--queries']
    if queries:
        with open(queries, 'r') as fp:
            queries = [query.strip() for query in fp]
    else:
        queries = average_precision.queries

    verbose = arguments['--verbose']

    hypothesis = arguments['<hypothesis.txt>']
    generator = average_precision(hypothesis)

    results = pd.DataFrame(index=queries, columns=average_precision.K + ['n'])

    if verbose:
        queries = tqdm(queries, unit='query')

    for query in queries:

        if verbose:
            queries.set_description('Querying "{query}"...'.format(query=query))

        values, n_relevant = generator.send(query)
        results.loc[query, average_precision.K] = values
        results.loc[query, 'n'] = n_relevant

    results.columns = ['AP@{k:d}'.format(k=k) for k in average_precision.K] + ['n_relevant']
    print(results.to_string(float_format=lambda f: "{f:.3f}".format(f=f)))

    mean_average_precision = results.mean()[:-1]
    mean_average_precision.index = ['MAP@{k:d}'.format(k=k) for k in average_precision.K]
    print('')
    print('MEAN AVERAGE PRECISION')
    print(mean_average_precision.to_string(float_format=lambda f: "{f:.3f}".format(f=f)))
