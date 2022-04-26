import numpy as np
import pyterrier as pt
import csv
from pyterrier.measures import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datetime import datetime

if not pt.started():
    pt.init()

dataset = pt.get_dataset('msmarco_passage')
corpus_path = dataset.get_corpus()[0]


def read_corpus(corpus_path):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("Process corpus")
    text_cospus = ""
    corpus = []
    with open(corpus_path, encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            corpus.append(row[1])
            text_cospus = text_cospus + str(row[1])
            line_count += 1
            if line_count % 50000 == 0:
                print(line_count)
            if line_count % 100000 == 0:
                break
        """   if line_count == 0:
                print("0"+ row[0]+ row[1] )
                print("\n")


        if line_count == 49:
            print("49"+ row[0]+ row[1] )
            print("\n")


        if line_count == 616:
            print("616"+ row[0]+ row[1] )
            print("\n")



        if line_count == 723:
            print("723"+ row[0]+ row[1] )
            print("\n")
            break"""
    print("The end")
    return {
        "corpus": corpus,
        "text_cospus": text_cospus,
    }


def statistics(corpus, text_corpus, top_k):
    print("\n\n")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    text_corpus = [text_corpus]

    tfidf_vector = TfidfVectorizer()
    tfidf = tfidf_vector.fit(corpus)

    bow_vecctorizeer = CountVectorizer()
    bow = bow_vecctorizeer.fit(corpus)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    bow_document_vector = bow.transform(text_corpus)
    tfidf_document_vector = tfidf.transform(text_corpus)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    sum_tf = np.sum(a=bow_document_vector, axis=0)
    sum_tfidf = np.sum(a=tfidf_document_vector, axis=0)
    sum_idf = tfidf.idf_

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    highest_tf = sum_tf.argsort().tolist()[0][::-1][:top_k]
    highest_idf = sum_idf.argsort()[::-1][:top_k]
    highest_tfidf = sum_tfidf.argsort().tolist()[0][::-1][:top_k]

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("Highest TF")
    total_words = sum(sum_tf.tolist()[0])

    for tf_index in highest_tf:
        for word, index in bow.vocabulary_.items():
            if index == tf_index:
                print(word, round(sum_tf.tolist()[0][tf_index] / total_words, 3))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("\n")
    print("Highest IDF")
    for tf_index in highest_idf:
        for word, index in tfidf.vocabulary_.items():
            if index == tf_index:
                print(word, round(sum_idf.tolist()[tf_index], 3))

    print("\n")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print("Highest TF-IDF")
    for tf_index in highest_tfidf:
        for word, index in tfidf.vocabulary_.items():
            if index == tf_index:
                print(word, round(sum_tfidf.tolist()[0][tf_index], 3))


#result = read_corpus(corpus_path)
#statistics(result["corpus"], result["text_cospus"], 11)
