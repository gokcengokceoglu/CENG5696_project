import re
import json
from nltk.corpus import stopwords


def lookup_query(inverted_index_data, query):
    """
    Returns the dictionary of terms with their correspondent Appearances.
    This is a very naive search since it will just split the terms and show
    the documents where they appear.
    """
    return {term: inverted_index_data[term] for term in query.split(' ') if term in inverted_index_data.keys()}

def highlight_term(id, term, text):
    replaced_text = text.replace(term, "\033[1;32;40m {term} \033[0;0m".format(term=term))
    return "--- document {id}: {replaced}".format(id=id, replaced=replaced_text)


def main():
    inverted_index = open('inverted_index.json')
    inverted_index_data = json.load(inverted_index)
    queries = open('queries.json')
    query_data = json.load(queries)
    database = open('database.json')
    db = json.load(database)
    # Search term can be changed, this is for example
    stop_words = set(stopwords.words('english'))
    matched = 0
    not_matched = 0
    for elm in query_data:
        search_term = elm['query']
        filtered_query = [w for w in search_term.split(' ') if not w.lower() in stop_words]
        filtered_query = ' '.join(filtered_query)
        print("Search term: {term}".format(term=filtered_query))
        result = lookup_query(inverted_index_data, filtered_query)
        docs = []
        for term in result.keys():
            docIds = [appearance['docId'] for appearance in result[term]]
            docs.append(docIds)
        flat_list = [item for sublist in docs for item in sublist]
        my_dict = {i: flat_list.count(i) for i in flat_list}
        sorted_dict = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        print(elm['song'])
        print("-----------------------------")
        print("Top 5 documents: ")
        matched_bool = False
        for item in sorted_dict:
            print(db.get(str(item[0]), None)['title'])
            print(item[1])
            if db.get(str(item[0]), None)['title'].lower() == elm['song'].lower():
                matched += 1
                matched_bool = True
            if matched_bool == False:
                not_matched += 1
                matched_bool = True
        print("-----------------------------")
    print("Matched: ", matched)
    print("Not matched: ", not_matched)



main()