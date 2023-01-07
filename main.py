import numpy as np
import pyterrier as pt
from sklearn import tree

from pyterrier.measures import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPRegressor
from sklearn import svm

from pipelineFunction import pipelineFunction

if not pt.started():
    pt.init()

pipelines = []
pipelinesNames = []
# dataset = pt.get_dataset('msmarco_passage')
dataset = pt.get_dataset('vaswani')

print("Files in msmarco_passage corpus: %s " % dataset.get_corpus())

# Get Index
# indexref_stemmed_docT5query = dataset.get_index("terrier_stemmed_docT5query")
# indexref_stemmed_deepct = dataset.get_index("terrier_stemmed_deepct")

indexref_stemmed_docT5query = dataset.get_index()

###
topics = dataset.get_topics()
qrels = dataset.get_qrels()

### Short
"""topics = dataset.get_topics("test-2019")
qrels = dataset.get_qrels("test-2019")"""
"""topics = dataset.get_topics("test-2020")
qrels = dataset.get_qrels("test-2020")"""

train_topics, val_topics, test_topics = np.split(topics, [int(.6 * len(topics)), int(.8 * len(topics))])
train_qrels, val_qrels, test_qrels = np.split(qrels, [int(.6 * len(qrels)), int(.8 * len(qrels))])

### Long
"""topics = dataset.get_topics("train")
qrels = dataset.get_qrels("train")

train_topics, val_topics = np.split(topics, [int(.8 * len(topics))])
train_qrels, val_qrels = np.split(qrels, [int(.8 * len(qrels))])

test_topics = dataset.get_topics("test-2019")
test_qrels = dataset.get_qrels("test-2019")"""
####


print(topics)
print(qrels)

print("Index Location:" + indexref_stemmed_docT5query)

# semelhante
# pipeline_tf = bm25 >> (tf_idf ** pl2)
pipeline_stemmed_docT5query = pt.FeaturesBatchRetrieve(indexref_stemmed_docT5query + "/data.properties", wmodel="BM25",
                                                       features=["WMODEL:Tf", "WMODEL:PL2"])
"""pipeline_stemmed_deepct = pt.FeaturesBatchRetrieve(indexref_stemmed_deepct + "/data.properties", wmodel="BM25",
                                                   features=["WMODEL:Tf", "WMODEL:PL2"])"""

pipelines.append(pipeline_stemmed_docT5query)
pipelinesNames.append("stemmed_docT5query")

# pipelines.append(pipeline_stemmed_deepct)
# pipelinesNames.append("stemmed_deepct")

d_topics = {
    "all": topics,
    "train": train_topics,
    "val": val_topics,
    "test": test_topics
}

d_qrels = {
    "all": qrels,
    "train": train_qrels,
    "val": val_qrels,
    "test": test_qrels
}

"""MNB = MultinomialNB()
pipelineFunction(pipelines, pipelinesNames, pt, MNB, d_topics, d_qrels, "NaiveBayes")"""
# dt =tree.DecisionTreeClassifier(random_state=0, min_samples_leaf=2)
# rf = RandomForestRegressor( ccp_alpha=0.015)
rf = RandomForestClassifier(random_state=0, n_estimators=100, criterion="gini", min_samples_leaf=2)
pipelineFunction(pipelines, pipelinesNames, pt, rf, d_topics, d_qrels, "RandomForest")
"""
mlp = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, learning_rate="adaptive", epsilon=1.0E-4,
                   max_iter=100, solver='sgd')
pipelineFunction(pipelines, pipelinesNames, pt, mlp, d_topics, d_qrels, "MLP")"""

"""svm = svm.SVC()
pipelineFunction(pipelines, pipelinesNames, pt, svm, d_topics, d_qrels, "SVM")"""

"""X = [[0, 0], [1, 1], [0,1]]
y = [0, 1,0]
clf = svm.SVC()
clf.fit(X, y)
result =clf.predict([[1, 1]])
print(result[0] )
"""
