import json
from spellchecker import SpellChecker
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
spell = SpellChecker()
f = open('lyrics.json')

data = json.load(f)
queries = []
df = pd.json_normalize(data)
for i in range(len(df)):
    misspelled = spell.unknown(df.loc[i, "lyrics"].split(" "))
    misspelled = [word for word in misspelled if word not in stop_words and word != '']
    for word in misspelled:
        # Get the one `most likely` answer
        corrected_word = spell.correction(word)
        if corrected_word is not None:
            df.loc[i, "lyrics"] = df.loc[i, "lyrics"].replace(word, corrected_word)

        # Get a list of `likely` options
        print(word, corrected_word)

