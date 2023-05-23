import gensim
from gensim.models import Word2Vec
import json


f = open('queries.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

words = []
for elm in data:
    query = elm['query'].split(" ")
    for q in query:
        words.append(q)
    song = elm['song'].split(" ")
    for s in song:
        words.append(s)


model1 = gensim.models.Word2Vec(words, min_count = 1,
                              vector_size = 100, window = 5)

