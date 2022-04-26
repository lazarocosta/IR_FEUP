from datetime import datetime


def printTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


def pipelineFunction(pipelines, pipelines_names, pt, clf, d_topics, d_qrels, file_location):
    tra_topic, val_topics, test_topics, topics = d_topics["train"], d_topics["val"], d_topics["test"], d_topics["all"]
    tra_qrels, val_qrels, test_qrels, qrels = d_qrels["train"], d_qrels["val"], d_qrels["test"], d_qrels["all"]

    #print(tra_topic)
    #print(tra_qrels)
    pipelinesRanked = []
    for pipeline in pipelines:
        newPipeline = pipeline >> pt.ltr.apply_learned_model(clf)
        print("Fit\n")
        newPipeline.fit(tra_topic, tra_qrels, val_topics, val_qrels)
        pipelinesRanked.append(newPipeline)

    printTime()

    # Experience
    print("Experiment:")
    dtExperiment = pt.Experiment(
        pipelinesRanked,
        test_topics,
        qrels,
        batch_size=200,
        filter_by_qrels=True,
        drop_unused=True,
        eval_metrics=[P @ 5, P @ 10, AP(rel=2), nDCG @ 10, nDCG @ 100, MRR],
        names=pipelines_names
    )
    print(dtExperiment)

    # Write Dataframe to file
    dtExperiment.to_csv(file_location + ".csv", sep=";")
