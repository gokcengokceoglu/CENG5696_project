
import nltk
nltk.download('punkt')
nltk.download('udhr')  # udhr = Universal Declaration of Human Rights
from nltk.corpus import udhr
import json
from nltk.stem.porter import PorterStemmer
import os
import re
from nltk.corpus import stopwords
import collections
import math
import typing
ps = PorterStemmer()
nltk.download('words')
nltk.download('webtext')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
webtext = set(nltk.corpus.webtext.words())
wordnet = set(nltk.corpus.wordnet.words())
words.update(webtext)
words.update(wordnet)


def extract_xgrams(text: str, n_vals: typing.List[int]) -> typing.List[str]:
    """
    Extract a list of n-grams of different sizes from a text.
    Params:
        text: the test from which to extract ngrams
        n_vals: the sizes of n-grams to extract
        (e.g. [1, 2, 3] will produce uni-, bi- and tri-grams)
    """
    xgrams = []

    for n in n_vals:
        # if n > len(text) then no ngrams will fit, and we would return an empty list
        if n < len(text):
            for i in range(len(text) - n + 1):
                ng = text[i:i + n]
                xgrams.append(ng)

    return xgrams

def build_model(text: str, n_vals: typing.List[int]) -> typing.Dict[str, int]:
    """
    Build a simple model of probabilities of xgrams of various lengths in a text
    Parms:
        text: the text from which to extract the n_grams
        n_vals: a list of n_gram sizes to extract
    Returns:
        A dictionary of ngrams and their probabilities given the input text
    """
    model = collections.Counter(extract_xgrams(text, n_vals))
    num_ngrams = sum(model.values())

    for ng in model:
        model[ng] = model[ng] / num_ngrams

    return model


def calculate_cosine(a: typing.Dict[str, float], b: typing.Dict[str, float]) -> float:
    """
    Calculate the cosine between two numeric vectors
    Params:
        a, b: two dictionaries containing items and their corresponding numeric values
        (e.g. ngrams and their corresponding probabilities)
    """
    numerator = sum([a[k]*b[k] for k in a if k in b])
    denominator = (math.sqrt(sum([a[k]**2 for k in a])) * math.sqrt(sum([b[k]**2 for k in b])))
    return numerator / (denominator + 0.00000001)

languages = ['english', 'german', 'dutch', 'french', 'italian', 'spanish', 'arabic', 'korean', 'chinese', 'turkish', 'russian']
language_ids = ['English-Latin1', 'German_Deutsch-Latin1', 'Dutch_Nederlands-Latin1', 'French_Francais-Latin1',
                'Italian_Italiano-Latin1',
                'Spanish_Espanol-Latin1', 'Arabic_Alarabia-Arabic', 'Korean_Hankuko-UTF8','Chinese_Mandarin-GB2312'
                ,'Turkish_Turkce-UTF8','Russian-UTF8']

raw_texts = {language: udhr.raw(language_id) for language, language_id in zip(languages, language_ids)}
# Build a model of each language
models = {language: build_model(text=raw_texts[language], n_vals=range(1, 4)) for language in languages}

def identify_language(
    text: str,
    language_models: typing.Dict[str, typing.Dict[str, float]],
    n_vals: typing.List[int]
    ) -> str:
    """
    Given a text and a dictionary of language models, return the language model
    whose ngram probabilities best match those of the test text
    Params:
        text: the text whose language we want to identify
        language_models: a Dict of Dicts, where each key is a language name and
        each value is a dictionary of ngram: probability pairs
        n_vals: a list of n_gram sizes to extract to build a model of the test
        text; ideally reflect the n_gram sizes used in 'language_models'
    """
    text_model = build_model(text, n_vals)
    language = ""
    max_c = 0
    for m in language_models:
        c = calculate_cosine(language_models[m], text_model)

        # The following line is just for demonstration, and can be deleted
        if c > max_c:
            max_c = c
            language = m
    return language

def remove_non_english_songs(dict):
    identified_language = identify_language(dict['lyrics'], models, n_vals=range(1, 4))
    dict['language'] = identified_language

    return dict

    '''
    sent = [w for w in nltk.wordpunct_tokenize(text) if w.lower() not in words]
    print(sent)
    '''

def remove_non_english_words(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return True
    return False

def modify_characters(dict):
    '''
    {"title": "10,000 miles",
    "album": "<single>",
    "release_date": "unidentified",
    "featured_artists": [],
    "producer_artists": [],
    "writer_artists": ["Cam\u2019ron"],
    "genius_track_id": 2958704,
    "genius_album_id": "none",
    "lyrics": "Lookin' up out my BenzKilla Cam nah nah nahKilla Cam nah nah nahI need to say sorry, y'all already knew itBut my pride won't let me do it, thoughHeard that, it's pride, won't let me do it thoughBest girl in the world, I might've blew itLike a small condom, I'll get through it thoughI'll get through it thoughHeard that, we can do it thoughI cheated, another one in the books for meShe clean, roll weed, and cook for meAin't hear from the kids, she would look for meNow she so mad that she won't even look at meDamn iPhones and cameras, nights in AtlantaTried to bring you gifts, she said, \"You ain't Santa\"I'm just showin' manners, she reachin' for the hammerShe ain't got ears, they replaced with antennasHard to believe I messed trust upNo good or bad, just tough luckI looked her in the eyes like, \"I fucked up\"Gripped up, my life had to monetizeMade mistakes, I apologizeDinner, movie, a date, you know, maybe thatSome dudes relate, I want my baby backI tell you a million times, I'm sorryWild life right here, no safariI'm dressed up, I 'fess upI messed up bad..."}
    :param dict:
    :return:
    '''
    title = dict['title']
    album = dict['album']
    release_date = dict['release_date']
    genius_track_id = dict['genius_track_id']
    genius_album_id = dict['genius_album_id']
    lyrics = dict['lyrics']
    title = re.sub(u"(\u2018|\u2019)", "'", title)
    album = re.sub(u"(\u2018|\u2019)", "'", album)
    lyrics = re.sub(u"(\u2018|\u2019)", "'", lyrics)
    lyrics = re.sub(u"(\u2005)", " ", lyrics)
    lyrics = re.sub(u"(\u00e9)", "e", lyrics)
    lyrics = re.sub(u"(\u0435)", "e", lyrics)
    lyrics = re.sub(u"(\u025b)", "e", lyrics)
    lyrics = re.sub(u"(\u00ef)", "i", lyrics)

    lyrics = re.sub(u"(\u00bd)", "", lyrics)
    lyrics = re.sub(u"(\u00e2)", "", lyrics)
    lyrics = re.sub(u"(\u205f)", " ", lyrics)
    lyrics = re.sub(u"(\u200b)", " ", lyrics)
    lyrics = re.sub(u"(\u00a0)", " ", lyrics)

    lyrics = re.sub(u"(\u00e1)", "a", lyrics)
    lyrics = re.sub(u"(\u00f1)", "n", lyrics)
    lyrics = re.sub(u"(\u00f6)", "n", lyrics)
    lyrics = re.sub(u"(\u00f8)", "o", lyrics)
    lyrics = re.sub(u"(\u00f3)", "o", lyrics)
    lyrics = re.sub(u"(\u0188)", "c", lyrics)
    lyrics = re.sub(u"(\u00f9)", "u", lyrics)


    lyrics = re.sub(u"(\u201d)", "", lyrics) # "
    lyrics = re.sub(u"(\u201c)", "", lyrics) # "
    lyrics = re.sub(u"(\u2026)", "", lyrics) # "
    lyrics = re.sub(u"(\u2605)", " ", lyrics) # BLACK STAR



    title = re.sub(u"(\u2013)", "", title).strip()
    album = re.sub(u"(\u2013)", "", album).strip()
    lyrics = re.sub(u"(\u2013)", "", lyrics).strip()
    #title = re.sub(r'[^\w\s]', '', title)
    #album = re.sub(r'[^\w\s]', '', album)
    lyrics = re.sub(r'[^\w\s]', '', lyrics).lower()
    lyrics = re.sub(r'\d', "", lyrics)
    dictionary = {"title": title,
                  "album": album,
                  "release_date": release_date,
                  "genius_track_id": genius_track_id,
                  "genius_album_id": genius_album_id,
                  "lyrics": lyrics}
    return dictionary

def remove_combined_words(text):
    words = text.split(" ")
    for word in words:
        res = re.search(r'[^ ][A-Z]\w*', word)
        if res:
            for i in range(len(word)):
                if i is not 0 and word[i].isupper():
                    new_word = word[:i] + " " + word[i:]
                    text = text.replace(word, new_word)
    return text


folder_path = './songs_by_artist_id'

file_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".json"):
            file_list.append(os.path.join(root, file))
data_list = []
for file in file_list:
    singer_info = []
    f = open(file)
    data = json.load(f)
    for keys, values in data.items():
        x = values
        if x['lyrics'] is '':
            continue
        x['lyrics'] = remove_combined_words(x['lyrics']).lower()
        x = modify_characters(x)
        x = remove_non_english_songs(x)
        if x['language'] is not 'english':
            continue
        if remove_non_english_words(x['lyrics']):
            continue
        x['artist_id'] = file.split('\\')[1][:-5]
        data_list.append(x)
with open("lyrics.json", "w") as f:
  json.dump(data_list, f, indent=4)


