import pyterrier as pt
from pyterrier.measures import *

if not pt.started():
    pt.init()

systems = []
names = []
dataset = pt.get_dataset('msmarco_passage')
# indexref = indexer.index(dataset.get_corpus())
index_ref_terrier_stemmed = dataset.get_index("terrier_stemmed")

# print(index_ref_terrier_stemmed.toString())

# load the index, print the statistics
index_terrier_stemmed = pt.IndexFactory.of(index_ref_terrier_stemmed)
print(index_terrier_stemmed.getCollectionStatistics().toString())

bm25_terrier_stemmed = pt.BatchRetrieve(index_terrier_stemmed, wmodel="BM25")
"""bm25 = pt.BatchRetrieve.from_dataset(
    "msmarco_passage", "terrier_stemmed", wmodel="BM25")"""
systems.append(bm25_terrier_stemmed)
names.append("bm25_terrier_stemmed")

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
print(topics)

print("Experiment:")
dtExperiment = pt.Experiment(
    systems,
    topics,
    qrels,
    batch_size=200,
    filter_by_qrels=True,
    drop_unused=True,
    eval_metrics=[P@5, P@10, AP(rel=2), nDCG@10, nDCG@100, MRR, MRR@10],
    round={"P@5": 3, "P@10": 3, "AP(rel=2)": 3, "nDCG@10": 3, "nDCG@100": 3, "MRR": 3, "MRR@10": 3},
    names=names
)
print(dtExperiment)

# Write Dataframe to file
dtExperiment.to_csv('default.csv', sep=";")
