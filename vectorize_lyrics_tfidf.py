import nltk
from nltk.probability import FreqDist
#nltk.download('punkt')
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
#import matplotlib
import json
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import math

def tokenize(text):
    print(text)
    print("---------------------")
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


import numpy as np


# compute Term frequency of a specific term in a document
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))


# IDF of a term
def inverseDocumentFrequency(term, documents):
    count = 0
    for doc in documents:
        if term.lower() in doc.lower().split():
            count += 1
    if count > 0:
        return 1.0 + math.log(float(len(documents)) / count)
    else:
        return 1.0

# tf-idf of a term in a document
def tf_idf(term, document, documents):
    tf = termFrequency(term, document)
    idf = inverseDocumentFrequency(term, documents)
    return tf * idf

def generateVectors(query, documents):
    tf_idf_matrix = np.zeros((len(query.split()), len(documents)))
    for i, s in enumerate(query.lower().split()):
        idf = inverseDocumentFrequency(s, documents)
        for j, doc in enumerate(documents):
            tf_idf_matrix[i][j] = idf * termFrequency(s, doc)
    return tf_idf_matrix

def word_count(s):
    counts = dict()
    words = s.lower().split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

def build_query_vector(query, documents):
    count = sum(word_count(query).values())
    vector = np.zeros((count, 1))
    for i, word in enumerate(query.lower().split()):
        vector[i] = float(word_count(query)[word]) / count * inverseDocumentFrequency(word, documents)
    return vector


def consine_similarity(v1, v2):
    return np.dot(v1, v2) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 0.000000001)


def compute_relevance(query_vector, documents, tf_idf_matrix, titles):
    max_similarity = []

    for i, doc in enumerate(documents):
        similarity = consine_similarity(tf_idf_matrix[:, i].reshape(1, len(tf_idf_matrix)), query_vector)
        max_similarity.append(similarity[0][0])
    sorted = np.argsort(max_similarity)[::-1]

    top_5 = sorted[:5]
    print("Titles of top songs from our dataset are using TF-IDF: ")
    for i in top_5:
        print("{}, similarity {}".format(titles[i], float(max_similarity[i])))
        print("-----------------------------")
