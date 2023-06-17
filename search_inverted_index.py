import json
from nltk.corpus import stopwords
import argparse
from vectorize_lyrics_tfidf import compute_relevance, generateVectors, build_query_vector
from spell_checker import spell_checker
from nltk.tokenize import RegexpTokenizer


def lookup_query(inverted_index_data, query):
    """
    Returns the dictionary of terms with their correspondent Appearances.
    This is a very naive search since it will just split the terms and show
    the documents where they appear.
    """
    return {term: inverted_index_data[term] for term in query.split(' ') if term in inverted_index_data.keys()}

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lyrics', type=str, default="")
    opt = parser.parse_args()
    return opt

def main(opt):
    inverted_index = open('inverted_index.json')
    inverted_index_data = json.load(inverted_index)
    database = open('database.json')
    tokenizer = RegexpTokenizer(r'\w+')
    db = json.load(database)
    # Search term can be changed, this is for example
    stop_words = set(stopwords.words('english'))
    search_term = opt.lyrics.replace("*", " ")
    search_term = spell_checker(search_term)
    print("-----------------------------")
    search_term = tokenizer.tokenize(search_term)

    filtered_query = [w.lower() for w in search_term if w.lower() not in stop_words]
    filtered_query = ' '.join(filtered_query)
    print("Your search term after removing stop words: {term}".format(term=filtered_query))
    print("-----------------------------")
    result = lookup_query(inverted_index_data, filtered_query)
    docs = []
    for term in result.keys():
        docIds = [appearance['docId'] for appearance in result[term]]
        docs.append(docIds)
    flat_list = [item for sublist in docs for item in sublist]
    my_dict = {i: flat_list.count(i) / len(filtered_query.split(" ")) for i in flat_list}
    sorted_dict = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)[:5]


    print("Titles of top songs from our dataset are using modified method: ")
    print("-----------------------------")
    for item in sorted_dict:
        print(db.get(str(item[0]), None)['title'])
        print("-----------------------------")

    f = open('lyrics.json')
    data = json.load(f)
    lyrics = []
    titles = []
    for elm in data:
        lyrics.append(elm['lyrics'])
        titles.append(elm['title'])

    tf_idf_matrix = generateVectors(filtered_query, lyrics)
    query_vector = build_query_vector(filtered_query, lyrics)
    compute_relevance(query_vector, lyrics, tf_idf_matrix, titles)

if __name__ == "__main__":
    opt = parse_opt()
    if opt.lyrics == "":
        print("Please input a lyric")
        exit()
    main(opt)



