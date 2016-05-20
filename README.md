# MediaEval Person Discovery 2016 Evaluation Metric

### Installation

```bash
$ git clone https://github.com/hbredin/PersonDiscovery2016Metric.git
$ cd PersonDiscovery2016Metric
$ pip install -r requirements.txt
```

### Command line

```bash
$ python metric.py --queries samples/queries.txt samples/reference.txt samples/hypothesis.txt
```

```
                   AP@1 AP@10 AP@100 n_relevant
nicolas_sarkozy   0.000 0.299  0.213        116
francois_hollande 0.000 0.083  0.150         26
alain_juppe       0.000 0.000  0.000          2

MEAN AVERAGE PRECISION
MAP@1     0.000
MAP@10    0.127
MAP@100   0.121
```

Option `--verbose` can be used to display a progress bar.

### API

```python
>>> from metric import AveragePrecision
>>> average_precision = AveragePrecision('samples/reference.txt', K=[1, 10, 100])
>>> generator = average_precision('samples/hypothesis.txt')
>>> (ap1, ap10, ap100), n_relevant = generator.send('nicolas_sarkozy')
```

### File formats

#### Reference

`corpus_id video_id shot_id person_name`
* `corpus_id` is the corpus identifier (`DW`, `INA` or `UPC`)
* `video_id` is the video identifier within the corpus
* `shot_id` is the shot identifier within the video
* `person_name` is the normalized name of the person


#### Hypothesis

`corpus_id video_id shot_id hypothesized_person_name confidence`
* `corpus_id` is the corpus identifier (`DW`, `INA` or `UPC`)
* `video_id` is the video identifier within the corpus
* `shot_id` is the shot identifier within the video
* `person_name` is the normalized name of the person
* `confidence` is a confidence score

If two (or more) people are visible AND speaking within a shot, both people should be returned:
```
corpus_id video_id shot_id hypothesized_person_name_1 confidence_1
corpus_id video_id shot_id hypothesized_person_name_2 confidence_2
```

### Mean average precision (implementation details)

Given a `person_name` query, hypotheses are sorted in decreasing order of the normalized edit distance between `person_name` and the `hypothesized_person_name`. Tied hypotheses are then sorted in decreasing order of their `confidence` score.
In the (unlikely) case there remain tied hypotheses, those are sorted by their `temporal rank` within each video.
