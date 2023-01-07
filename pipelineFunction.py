from datetime import datetime
from sklearn import tree
import matplotlib.pyplot as plt
from pyterrier.measures import *


def printTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


def pipelineFunction(pipelines, pipelines_names, pt, clf, d_topics, d_qrels, file_location):
    tra_topic, val_topics, test_topics, topics = d_topics["train"], d_topics["val"], d_topics["test"], d_topics["all"]
    tra_qrels, val_qrels, test_qrels, qrels = d_qrels["train"], d_qrels["val"], d_qrels["test"], d_qrels["all"]

    # print(tra_topic)
    # print(tra_qrels)
    pipelines_ranked = []
    for pipeline in pipelines:
        new_pipeline = pipeline >> pt.ltr.apply_learned_model(clf)
        print("Fit\n")
        new_pipeline.fit(tra_topic, tra_qrels, val_topics, tra_qrels)

        plt.figure(figsize=(20, 20))
        este = clf.estimators_[0].tree_.feature

        #tree.plot_tree(clf, filled=True, max_depth=3, fontsize=8)
        tree.plot_tree(clf.estimators_[0], filled=True, max_depth=2, fontsize=10)

        plt.title(file_location)
        # plt.show()
        plt.savefig(file_location + ".png")
        pipelines_ranked.append(new_pipeline)

        printTime()

        # Experience
        print("Experiment:")
        # dt_experiment = pt.Experiment(
        #     pipelines_ranked,
        #     test_topics,
        #     test_qrels,
        #     batch_size=200,
        #     filter_by_qrels=True,
        #     drop_unused=True,
        #     eval_metrics=[P @ 5, P @ 10, AP(rel=2), nDCG @ 10, nDCG @ 100, MRR, MRR @ 10],
        #     round={"P@5": 3, "P@10": 3, "AP(rel=2)": 3, "nDCG@10": 3, "nDCG@100": 3, "RR": 3, "RR@10": 3},
        #     names=pipelines_names
        #
        # )
        # print(dt_experiment)

        # Write Dataframe to file
        # dt_experiment.to_csv("./results/" + file_location + ".csv", sep=";")
