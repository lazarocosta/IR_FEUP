import uuid
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
if not pt.started():
    pt.init()

systems = []
names = []
dataset = pt.get_dataset('msmarco_passage')

bm25 = pt.BatchRetrieve.from_dataset(
    "msmarco_passage", "terrier_stemmed", wmodel="BM25")
systems.append(bm25)
names.append("bm25")

bm25deepct = pt.BatchRetrieve.from_dataset(
    "msmarco_passage", "terrier_stemmed_deepct", wmodel="BM25")
systems.append(bm25deepct)
names.append("stemmed_deepct")

bm25_stemmed_docT5query = pt.BatchRetrieve.from_dataset(
    'msmarco_passage', 'terrier_stemmed_docT5query', wmodel='BM25')
systems.append(bm25_stemmed_docT5query)
names.append("bm25_stemmed_docT5query")

bm25_terrier_stemmed_text = pt.BatchRetrieve.from_dataset(
    'msmarco_passage', 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
systems.append(bm25_terrier_stemmed_text)
names.append("bm25_stemmed_text")


topics = dataset.get_topics("test-2019")
qrels = dataset.get_qrels("test-2019")

print("Experiment:")
dtExperiment = pt.Experiment(
    systems,
    topics,
    qrels,
    batch_size=200,
    filter_by_qrels=True,
    drop_unused=True,
    eval_metrics=[P@5, P@10, AP(rel=2), nDCG@5, nDCG@10],
    names=names
)

print(dtExperiment)
