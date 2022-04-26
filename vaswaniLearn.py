import uuid
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

if not pt.started():
    pt.init()


##Get dataset
dataset = pt.get_dataset("vaswani") # funciona 10M
#dataset = pt.get_dataset('msmarco_passage') #900M funciona

#topics = dataset.get_topics('train')
#qrels = dataset.get_qrels()



print("Files in vaswani corpus: %s " % dataset.get_corpus())
index_path = "./index/index" + str(uuid.uuid1())

# build the index
#indexer = pt.TRECCollectionIndexer(index_path, verbose=True, blocks=False)
# this downloads the file msmarco-docs.trec.gz
#indexref = indexer.index(dataset.get_corpus())
indexref = dataset.get_index()

#print(indexref.toString())

# load the index, print the statistics
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())

tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")


pipeline_tf = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])
pipeline_tfidf = bm25 >> (tf_idf ** pl2)



####
###                SKLearn

from sklearn.ensemble import RandomForestRegressor
import numpy as np

indexref = dataset.get_index()
topics = dataset.get_topics()
qrels = dataset.get_qrels()

train_topics, valid_topics, test_topics = np.split(topics, [int(.6*len(topics)), int(.8*len(topics))])

rf = RandomForestRegressor(n_estimators=200)
rf_pipe_tf = pipeline_tf >> pt.ltr.apply_learned_model(rf)
rf_pipe_tf.fit(dataset.get_topics(), dataset.get_qrels())

rf_pipe_tf_idf = pipeline_tfidf >> pt.ltr.apply_learned_model(rf)
rf_pipe_tf_idf.fit(dataset.get_topics(), dataset.get_qrels())

##rf_both = RandomForestRegressor(n_estimators=400, max_depth=10)
#rf_both_pipe = pipeline >> pt.ltr.apply_learned_model(rf_both)
#rf_both_pipe.fit(dataset.get_topics(), dataset.get_qrels())
###PIOR

#rf_maxdepth = RandomForestRegressor(max_depth=10)
#rf_maxdepth_pipe = pipeline >> pt.ltr.apply_learned_model(rf_maxdepth)
#rf_maxdepth_pipe.fit(dataset.get_topics(), dataset.get_qrels())
###PIOR

#bm25
#PIOR



print('Experiment:')
dtExperiment = pt.Experiment([ rf_pipe_tf, rf_pipe_tf_idf],
    test_topics,
    qrels, 
    eval_metrics=[P@5, P@10, "map", nDCG@5, nDCG@10], 
    names=["rf_pipe_tf", "rf_pipe_tf_idf"],
    #save_dir="./sklearn/"
    )

print(dtExperiment)
