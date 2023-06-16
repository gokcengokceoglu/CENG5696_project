#python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as g
import logging
# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

# Reads ‘alice.txt’ file
import json
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import plotly.graph_objects as go

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tqdm
ps = PorterStemmer()

nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

from gensim.models.callbacks import CallbackAny2Vec
class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

def get_dataset(query_file, lyrics_file):
    f_query = open(query_file)
    query_data = json.load(f_query)
    f_lyric = open(lyrics_file)
    lyric_data = json.load(f_lyric)
    sentences = []
    lyrics = []
    ids = []
    for elm in query_data:
        query = elm['query']
        word_tokens = word_tokenize(query)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        sentences.append(filtered_sentence)
    for elm in lyric_data:
        lyric = elm['lyrics']
        sentences.append([lyric])
        lyrics.append(lyric)
        ids.append(elm['genius_track_id'])
    return sentences, lyrics, ids

def get_song_vectors(model, lyrics_file, vector_size = 100):
    song_word2vec_embeddings = []
    f_lyric = open(lyrics_file)
    lyric_data = json.load(f_lyric)
    for elm in tqdm.tqdm(lyric_data):
        num_words = 0
        song_vector = np.zeros((vector_size,))
        lyric = elm['lyrics']
        title = elm['title']
        word_tokens = word_tokenize(lyric)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        #filtered_sentence = [ps.stem(w) for w in filtered_sentence]
        for word in filtered_sentence:
            vect = model.wv.get_vector(word)
            song_vector += vect
            num_words += 1
        mean_vector = song_vector / num_words
        song_word2vec_embeddings.append([title, list(mean_vector)])
    with open("song_doc2vec_embeddings.json", "w") as fp:
        json.dump(song_word2vec_embeddings, fp)
    return

def calc_cosine_sim(A,B):
    cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cosine

def sort_sims(list):
    list.sort(key = lambda x: x[1])
    return list

def rank_songs(model, query, vector_size=100):
    num_words = 0
    query_vector = np.zeros((vector_size,))
    word_tokens = word_tokenize(query)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = [ps.stem(w) for w in filtered_sentence]
    for word in filtered_sentence:
        vect = model.wv.get_vector(word)
        query_vector += vect
        num_words += 1
    mean_query_vector = query_vector / num_words
    song_w2v = open("song_doc2vec_embeddings.json")
    song_embeddings = json.load(song_w2v)
    sims = []
    for elm in song_embeddings:
        title = elm[0]
        vect = np.asarray(elm[-1])
        sim = calc_cosine_sim(vect, mean_query_vector)
        sims.append([title,sim])
    ranked_songs = sort_sims(sims)
    return ranked_songs

def test(query_file,model):
    file1 = open("myfile.txt","w+")
    f_query = open(query_file)
    query_data = json.load(f_query)
    for elm in query_data:
        query = elm['query']
        song = elm['song']
        ranked_songs = rank_songs(model, query, vector_size=100)
        first_songs = ranked_songs[-5:-1]
        file1.write(str(song) + "\n")
        for f in first_songs:
            file1.write(str(f) + "\n")
            file1.write("-------------------\n")
    file1.close()

query_file = "queries.json"
lyrics_file = "lyrics.json"

'''
ds, lyrics, ids = get_dataset(query_file, lyrics_file)
documents = [g.doc2vec.TaggedDocument(words=doc.split(' '),tags= [id]) for doc, id in zip(lyrics, ids)]
sentences = [g.doc2vec.TaggedDocument(words=doc, tags=[id]) for doc, id in zip(ds, ids)]
documents = sentences + documents
model = gensim.models.Doc2Vec( min_count = 1, vector_size = 100,
                                            window = 5, callbacks=[callback()])
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=10)
model.save("d2v.model")
'''
model = gensim.models.Doc2Vec.load("d2v.model")
similar_doc = model.docvecs.most_similar(0)

get_song_vectors(model, lyrics_file, vector_size = 100)
test(query_file, model)
