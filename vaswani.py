import uuid
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
if not pt.started():
    pt.init()


##Get dataset
dataset = pt.get_dataset("vaswani") # funciona 10M
#dataset = pt.get_dataset("trec-deep-learning-docs")  #7.9G funciona == msmarco_document
#dataset = pt.get_dataset('msmarco_document') #7.9G funciona

#dataset = pt.get_dataset('trec-deep-learning-passages') #900M funciona ==msmarco_passage
#dataset = pt.get_dataset('msmarco_passage') #900M funciona

#topics = dataset.get_topics('train')
#qrels = dataset.get_qrels()



print("Files in vaswani corpus: %s " % dataset.get_corpus())

# build the index
#indexer = pt.TRECCollectionIndexer(index_path, verbose=True, blocks=False)
# this downloads the file msmarco-docs.trec.gz
#indexref = indexer.index(dataset.get_corpus())
#indexref = dataset.get_index()

#print(indexref.toString())

# load the index, print the statistics
index = pt.IndexFactory.of(dataset.get_index())
print(index.getCollectionStatistics().toString())

tf = pt.BatchRetrieve(index, wmodel="Tf")
tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")
pipeline = bm25 >> (tf_idf ** pl2)


#pipeline2 = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])


print("Experiment:")

dtExperiment = pt.Experiment(
    [ tf, tf_idf, bm25, pl2, pipeline],
    dataset.get_topics(),
    dataset.get_qrels(),
    #eval_metrics=[P @ 5, P @ 10, AP(rel=2), nDCG @ 10, nDCG @ 100, MRR, MRR @ 10],
    eval_metrics=["map", "recip_rank"],
    #round={"P@5": 3, "P@10": 3, "AP(rel=2)": 3, "nDCG@10": 3, "nDCG@100": 3, "RR": 3, "RR@10": 3},
    perquery=True
   )

print(dtExperiment)
dtExperiment.to_csv('./results/vaswani.csv', sep=";")




#dtBatch=pt.BatchRetrieve(indexref).search("mathematical")
#topics = pd.DataFrame([["2", "experimental results"]],columns=['qid','query'])
#dtBatch= dtBatch.head(10)

#print(dtBatch)



####
###                SKLearn

"""from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=400)
rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)
rf_pipe.fit(dataset.get_topics(), dataset.get_qrels())
pt.Experiment([bm25, rf_pipe], test_topics, dataset.get_qrels(), ["map"], names=["BM25 Baseline", "LTR"])"""



#####
###                    GRADIENT BOOST
"""import xgboost as xgb
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

pt.Experiment(
    [bm25, lmart_x_pipe, lmart_l_pipe],
    test_topics,
    test_qrels,
    ["map"],
    names=["BM25 Baseline", "LambdaMART (xgBoost)", "LambdaMART (LightGBM)" ]
)"""
