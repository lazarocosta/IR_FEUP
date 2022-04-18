import os
import uuid
import pandas as pd
from tabulate import tabulate
import pyterrier as pt
if not pt.started():
      pt.init()


##Get dataset
dataset = pt.get_dataset("vaswani")
print("Files in vaswani corpus: %s " % dataset.get_corpus())
index_path = "./index/index" + str(uuid.uuid1())

# build the index
indexer = pt.TRECCollectionIndexer(index_path, verbose=True, blocks=False)
# this downloads the file msmarco-docs.trec.gz
indexref = indexer.index(dataset.get_corpus())
print(indexref.toString())

# load the index, print the statistics
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())

tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")

print("Experiment:")
dtExperiment = pt.Experiment(
    [tf_idf, bm25, pl2],
    dataset.get_topics(),
    dataset.get_qrels(), 
    eval_metrics=["P.5", "P.10", "ndcg_cut.10", "map"],
    round={"map" : 3},)
    #baseline=0)

print(dtExperiment)
print("\n\n")



dtBatch=pt.BatchRetrieve(indexref).search("mathematical")
#topics = pd.DataFrame([["2", "experimental results"]],columns=['qid','query'])
dtBatch= dtBatch.head(10)

print(dtBatch)


