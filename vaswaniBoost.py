import uuid
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
import numpy as np


if not pt.started():
    pt.init()


##Get dataset
dataset = pt.get_dataset('vaswani') #900M funciona ==msmarco_passage
#dataset = pt.get_dataset('trec-deep-learning-passages') #900M funciona ==msmarco_passage


#dataset = pt.get_dataset('msmarco_passage') #900M funciona
print("Files in msmarco_passage corpus: %s " % dataset.get_corpus())
index_path = "./index/index" + str(uuid.uuid1())

# build the index
indexer = pt.TRECCollectionIndexer(index_path, verbose=True, blocks=False)
indexref = indexer.index(dataset.get_corpus())
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())

topics = dataset.get_topics()
qrels = dataset.get_qrels()


train_topics, validation_topics, test_topics = np.split(topics, [int(.6*len(topics)), int(.8*len(topics))])
train_qrels, validation_qrels, test_qrels = np.split(qrels, [int(.6*len(topics)), int(.8*len(topics))])

tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")
pipeline = bm25 >> (tf_idf ** pl2)



#####
###                    GRADIENT BOOST
import xgboost as xgb
# this configures XGBoost as LambdaMART
lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
      learning_rate=0.1,
      gamma=1.0,
      min_child_weight=0.1,
      max_depth=6,
      verbose=2,
      random_state=42)

lmart_x_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
lmart_x_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)

import lightgbm as lgb
# this configures LightGBM as LambdaMART
lmart_l = lgb.LGBMRanker(task="train",
    min_data_in_leaf=1,
    min_sum_hessian_in_leaf=100,
    max_bin=255,
    num_leaves=7,
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[1, 3, 5, 10],
    learning_rate= .1,
    importance_type="gain",
    num_iterations=10)
lmart_l_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
lmart_l_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)

experiment =pt.Experiment(
    [bm25, lmart_x_pipe, lmart_l_pipe, tf_idf, pl2, pipeline],
    test_topics,
    test_qrels,
    eval_metrics=[P@5, P@10, "map", nDCG@5, nDCG@10],
    names=["BM25", "xgBoost", "LightGBM", "tf_idf", "pl2", "pipeline"]
)
print(experiment)
