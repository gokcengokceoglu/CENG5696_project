import glob, os, shutil
import json
import re
import tqdm

file1 = open('queries.txt', 'r')
lines = file1.readlines()
regex = re.compile(r'#.{0,5}\.')

dicts = []


for l in tqdm.tqdm(lines):
    try:
        l = regex.sub(' ', l)

        strings = re.split(r'[–]', l)

        # Get the first string
        query_string = strings[0]

        # Get the second string
        singer_song_string = strings[1]
        singer_song_strings = re.split(r'[,]', singer_song_string)
        singer = singer_song_strings[0]
        song = singer_song_strings[1]

        query_string = re.sub(u"(\u2018|\u2019)", "'", query_string)
        singer = re.sub(u"(\u2018|\u2019)", "'", singer)
        song = re.sub(u"(\u2018|\u2019)", "'", song)
        query_string = re.sub(u"(\u2013)", "", query_string).strip()
        singer = re.sub(u"(\u2013)", "", singer).strip()
        song = re.sub(u"(\u2013)", "", song).strip()
        query_string = re.sub(r'[^\w\s]','',query_string).lower()
        song = re.sub(r'[^\w\s]','',song).lower()
        singer = re.sub(r'[^\w\s]','',singer).lower()

        dictionary = {"query": query_string, "song": song ,"singer": singer}
        dicts.append(dictionary)

    except:
       continue



with open("queries.json", "w") as f:
  json.dump(dicts, f, indent=4)
