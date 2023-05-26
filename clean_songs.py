import nltk
nltk.download('punkt')
import json
from nltk.stem.porter import PorterStemmer
import os
import re
from nltk.corpus import stopwords
import tqdm
ps = PorterStemmer()

nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
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
        x['lyrics'] = remove_combined_words(x['lyrics']).lower()
        x = modify_characters(x)
        x['artist_id'] = file.split('\\')[1][:-5]
        data_list.append(x)
with open("lyrics.json", "w") as f:
  json.dump(data_list, f, indent=4)


