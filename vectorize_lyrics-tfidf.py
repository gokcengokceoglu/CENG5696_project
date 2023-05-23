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

def tokenize(text):
    print(text)
    print("---------------------")
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

f = open('queries.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
# font = {'family' : 'normal',
#         'size'   : 12}
queries = []
#matplotlib.rc('font', **font)
df = pd.json_normalize(data)
for i in range(len(df)):
    #print(cultural_discourse_type)
    if (df.loc[i, "query"] != ''):
        queries.append(df.loc[i, "query"])


from sklearn.feature_extraction.text import TfidfVectorizer

# list of text documents

# create the transform
vectorizer = TfidfVectorizer(stop_words='english')
# tokenize and build vocab
model = vectorizer.fit_transform(queries)
print(model)
print(pd.DataFrame(model.toarray(), columns = vectorizer.get_feature_names()))
