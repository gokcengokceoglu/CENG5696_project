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
import string

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
    for elm in query_data:
        query = elm['query']
        word_tokens = word_tokenize(query)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = [ps.stem(w) for w in filtered_sentence]
        sentences.append(filtered_sentence)
    for elm in lyric_data:
        lyric = elm['lyrics']
        word_tokens = word_tokenize(lyric)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = [ps.stem(w) for w in filtered_sentence]
        sentences.append(filtered_sentence)
    return sentences




def train_word2vec(words, model_type="skipgram", vector_size = 100, window = 5, epochs = 10):

    if (model_type == "skipgram"):
         model = gensim.models.Word2Vec(words, min_count = 1, vector_size = vector_size,
                                            window = window, sg = 1, callbacks=[callback()])
    else: # Train C-BOW Model
        model = gensim.models.Word2Vec(words, min_count = 1,
                                        vector_size = vector_size, window = window, epochs=epochs, callbacks=[callback()])
    return model
    
  
def visualize_wordmap(model):
    X = []
    ii = 0
    words_vis = []
    for word in model.wv.index_to_key:
        ii+=1
        print(word)
        if (len(word) < 5):
            continue
        #X.append(word)
        X.append(model.wv.get_vector(word))
        words_vis.append(word)
        if (ii > 200):
            break

    #X = model1[model1.wv.index_to_key]
    pca = PCA(n_components=2)

    result = pca.fit_transform(X)

    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1], s=3)
    words = list(model.wv.index_to_key)

    for i, word in enumerate(words_vis):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=8)

    plt.show()


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
        filtered_sentence = [ps.stem(w) for w in filtered_sentence]
        for word in filtered_sentence:
            vect = model.wv.get_vector(word)
            song_vector += vect
            num_words += 1
        mean_vector = song_vector / num_words
        title = title.translate(str.maketrans('', '', string.punctuation)).lower()
        song_word2vec_embeddings.append([title, list(mean_vector)])
    with open("song_word2vec_embeddings.json", "w") as fp:
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
    song_w2v = open("song_word2vec_embeddings.json")
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
    file1 = open("test_results.txt","w+")
    f_query = open(query_file)
    query_data = json.load(f_query)
    correct_cls = 0
    false_cls = 0
    for elm in query_data:
        query = elm['query']
        song = elm['song']
        ranked_songs = rank_songs(model, query, vector_size=300)
        song_names_ranked = [s[0] for s in ranked_songs]
        try:
            print(song_names_ranked.index(song))
        except:
            print(song + " does not exist!!")
            false_cls = false_cls -1
        first_songs = ranked_songs[-15:-1]
        file1.write(str(song) + "\n")
        for f in first_songs:
            file1.write(str(f) + "\n")
        file1.write("-------------------\n")
        for results_song in first_songs:
            if (results_song[0] == song):
                correct_cls += 1
        false_cls += 1
    print(correct_cls)
    print(false_cls)
    file1.close()


query_file = "queries.json"
lyrics_file = "lyrics.json"

ds = get_dataset(query_file, lyrics_file)
model = train_word2vec(ds, vector_size = 300, window = 3, epochs = 10)
get_song_vectors(model, lyrics_file, vector_size = 300)
test(query_file, model)
#visualize_wordmap(model)
