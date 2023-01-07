import pyterrier as pt
from pyterrier.measures import *

if not pt.started():
    pt.init()

systems = []
names = []
datasetName = "msmarco_passage"
dataset = pt.get_dataset(datasetName)

# GET Index
index_ref_terrier_stemmed = dataset.get_index("terrier_stemmed")
index_ref_stemmed_deepct = dataset.get_index("terrier_stemmed_deepct")
index_ref_stemmed_docT5query = dataset.get_index("terrier_stemmed_docT5query")
index_ref_stemmed_text = dataset.get_index("terrier_stemmed_text")

# print(index_ref_terrier_stemmed.toString())
# load the index, print the statistics
index_terrier_stemmed = pt.IndexFactory.of(index_ref_terrier_stemmed)
index_stemmed_deepct = pt.IndexFactory.of(index_ref_stemmed_deepct)
index_stemmed_docT5query = pt.IndexFactory.of(index_ref_stemmed_docT5query)
index_stemmed_text = pt.IndexFactory.of(index_ref_stemmed_text)

print(index_terrier_stemmed.getCollectionStatistics().toString())

## Get Pipeline
"""bm25 = pt.BatchRetrieve.from_dataset(
    datasetName, "terrier_stemmed", wmodel="BM25")"""
bm25_terrier_stemmed = pt.BatchRetrieve(index_terrier_stemmed, wmodel="BM25")
systems.append(bm25_terrier_stemmed)
names.append("terrier_stemmed")

bm25_stemmed_deepct = pt.BatchRetrieve(index_stemmed_deepct, wmodel="BM25")
systems.append(bm25_stemmed_deepct)
names.append("terrier_stemmed_deepct")

bm25_stemmed_docT5query = pt.BatchRetrieve(index_stemmed_docT5query, wmodel="BM25")
systems.append(bm25_stemmed_docT5query)
names.append("terrier_stemmed_docT5query")

bm25_stemmed_text = pt.BatchRetrieve(index_stemmed_text, wmodel="BM25", metadata=['docno', 'text'])
systems.append(bm25_stemmed_text)
names.append("terrier_stemmed_text")

tf = pt.BatchRetrieve(index_ref_stemmed_docT5query, wmodel="Tf")
tf_idf = pt.BatchRetrieve(index_ref_stemmed_docT5query, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index_ref_stemmed_docT5query, wmodel="BM25")
pl2 = pt.BatchRetrieve(index_ref_stemmed_docT5query, wmodel="PL2")
pipeline = bm25 >> (tf_idf ** pl2)

pipeline1 = bm25 >> (tf ** pl2)
pipeline2 = bm25 >> (tf ** pl2 ** tf_idf)

pipeline3 = bm25 >> (tf_idf)
pipeline4 = bm25 >> (pl2)
pipeline5 = bm25 >> (tf)

systems.append(pipeline)
names.append("pipeline")

systems.append(pipeline1)
names.append("pipeline1")

systems.append(pipeline2)
names.append("pipeline2")

systems.append(pipeline3)
names.append("pipeline3")

systems.append(pipeline4)
names.append("pipeline4")

systems.append(pipeline5)
names.append("pipeline5")

topics = dataset.get_topics("test-2019")
qrels = dataset.get_qrels("test-2019")
# print(topics)
# print(qrels)


print("Experiment:")
dtExperiment = pt.Experiment(
    systems,
    topics,
    qrels,
    batch_size=200,
    filter_by_qrels=True,
    drop_unused=True,
    #eval_metrics=[P @ 5, P @ 10,AP, AP(rel=2), nDCG @ 10, nDCG @ 100, MRR, MRR @ 10],
    eval_metrics=["map"],

    #round={"P@5": 3, "P@10": 3, "AP":3, "AP(rel=2)": 3, "nDCG@10": 3, "nDCG@100": 3, "RR": 3, "RR@10": 3},
    #names=names,
    perquery=True

)
print(dtExperiment)

# Write Dataframe to file
dtExperiment.to_csv('./results/default2.csv', sep=";")
