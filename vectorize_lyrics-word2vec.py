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
ps = PorterStemmer()

nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

f = open('queries.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

words = []
for elm in data:
    query = elm['query']
    word_tokens = word_tokenize(query)
    # converts the words in word_tokens to lower case and then checks whether
    #they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = [ps.stem(w) for w in filtered_sentence]
    #with no lower case conversion
    words.append(filtered_sentence)
    print(filtered_sentence)
        # song = elm['song'].split(" ")
        #     words.append(s)


# Create CBOW model
# model1 = gensim.models.Word2Vec(words, min_count = 1,
# 							vector_size = 100, window = 5, epochs=50)

model1 = gensim.models.Word2Vec(words, min_count = 1, vector_size = 100,
										window = 5, sg = 1)
X = []
ii = 0
words_vis = []
for word in model1.wv.index_to_key:
    ii+=1
    print(word)
    if (word == "shot"):
        continue
    #X.append(word)
    X.append(model1.wv.get_vector(word))
    words_vis.append(word)
    if (ii > 100):
        break


#X = model1[model1.wv.index_to_key]
pca = PCA(n_components=2)

result = pca.fit_transform(X)

# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1], s=3)
words = list(model1.wv.index_to_key)

for i, word in enumerate(words_vis):
   plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=8)

plt.show()
# # Print results
# print("Cosine similarity between 'alice' " +
# 			"and 'wonderland' - CBOW : ",
# 	model1.wv.similarity('stop', 'just'))
	
# # print("Cosine similarity between 'alice' " +
# # 				"and 'machines' - CBOW : ",
# # 	model1.wv.similarity('alice', 'machines'))

# # Create Skip Gram model
# model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
# 											window = 5, sg = 1)

# # Print results
# print("Cosine similarity between 'alice' " +
# 		"and 'wonderland' - Skip Gram : ",
# 	model2.wv.similarity('stop', 'just'))
	
# # print("Cosine similarity between 'alice' " +
# # 			"and 'machines' - Skip Gram : ",
# # 	model2.wv.similarity('alice', 'machines'))
