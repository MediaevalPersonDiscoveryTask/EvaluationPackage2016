# MediaEval Person Discovery 2016 Evaluation Metric

visible AND speaking

### Usage

```python
>>> from metric import AveragePrecision
>>> average_precision = AveragePrecision('samples/reference.txt', K=[1, 10, 100, 1000])
>>> generator = average_precision('samples/hypothesis.txt')
>>> TEMPLATE = '{query:20s} | Average Precision @ {k:4d} | {value:.3f}'
>>> for query in ['nicolas_sarkozy', 'francois_hollande', 'alain_juppe']:
...     ap, n_relevant = generator.send(query)
...     for i, k in enumerate(average_precision.K):
...         print(TEMPLATE.format(query=query, k=k, value=ap[i]))
```
```
nicolas_sarkozy      | Average Precision @    1 | 0.000
nicolas_sarkozy      | Average Precision @   10 | 0.299
nicolas_sarkozy      | Average Precision @  100 | 0.213
nicolas_sarkozy      | Average Precision @ 1000 | 0.194
francois_hollande    | Average Precision @    1 | 0.000
francois_hollande    | Average Precision @   10 | 0.083
francois_hollande    | Average Precision @  100 | 0.150
francois_hollande    | Average Precision @ 1000 | 0.150
alain_juppe          | Average Precision @    1 | 0.000
alain_juppe          | Average Precision @   10 | 0.000
alain_juppe          | Average Precision @  100 | 0.000
alain_juppe          | Average Precision @ 1000 | 0.002
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

### Mean average precision

Given a `person_name` query, hypotheses are sorted in decreasing order of the normalized edit distance between `person_name` and the `hypothesized_person_name`. Tied hypotheses are then sorted in decreasing order of their `confidence` score.
In the (unlikely) case there remain tied hypotheses, those are sorted by their `temporal rank` within each video.
