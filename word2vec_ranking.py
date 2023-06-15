import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import scipy
from scipy import spatial
from nltk.tokenize.toktok import ToktokTokenizer
import re
import pandas as pd
import os
import json
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

f = open('lyrics.json')
data = json.load(f)

q = open('queries.json')
queries = json.load(q)

words = []
doc_dict = {}
for elm in data:
    query = elm['lyrics']
    word_tokens = word_tokenize(query)
    # converts the words in word_tokens to lower case and then checks whether
    #they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stopword_list and len(w) >= 2]
    filtered_sentence = [ps.stem(w) for w in filtered_sentence]
    #with no lower case conversion
    d1 = {elm['title']: (filtered_sentence)}
    doc_dict.update(d1)
    words.append(filtered_sentence)

model1 = gensim.models.Word2Vec(words, min_count=1, vector_size=300, window=5, sg=1)
def embeddings(word):
    if word in model1.wv.key_to_index:
        return model1.wv.get_vector(word)
    else:
        return np.zeros(300)

out_dict = {}
for k,v in doc_dict.items():
    try:
        average_vector = (np.mean(np.array([embeddings(x) for x in v]), axis=0))
    except:
        average_vector = np.zeros(100)
    d1 = {k: (average_vector)}
    out_dict.update(d1)

def get_sim(query_embedding, average_vec):
    sim = [(1 - scipy.spatial.distance.cosine(query_embedding, average_vec))]
    return sim

def rankings(query):
    query_words = (np.mean(np.array([embeddings(x) for x in nltk.word_tokenize(query.lower())], dtype=float), axis=0))
    rank = []
    for k, v in out_dict.items():
        rank.append((k, get_sim(query_words, v)))
    rank = sorted(rank, key=lambda t: t[1], reverse=True)
    print("Ranked documents: ")
    return rank[:30]

for q in queries:
    print(q)
    print(rankings(q['query']))
    print("\n")