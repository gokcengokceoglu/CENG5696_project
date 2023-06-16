import json , codecs
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

import numpy as np

import tqdm

import torch
from transformers import BertTokenizer, BertModel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                                                     


class CustomSongDataset(torch.utils.data.Dataset):

    def __init__(self, lyrics, titles, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.lyrics = lyrics
        self.titles = titles
        self.max_len = max_len
    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        lyrics = str(self.lyrics[index])
        lyrics = " ".join(lyrics.split())
        inputs = self.tokenizer.encode_plus(
            lyrics,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'titles':self.titles[index]
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        return output

model = BERTClass()
model.to(device)


def get_dataset(query_file, lyrics_file):
    f_query = open(query_file)
    query_data = json.load(f_query)
    f_lyric = open(lyrics_file)
    lyric_data = json.load(f_lyric)
    lyrics = []
    lyric_titles = []
    queries = []
    query_titles = []

    for elm in query_data:
        query = elm['query']
        song_title = elm["song"]
        queries.append(query)
        query_titles.append(song_title)

    for elm in lyric_data:
        lyric = elm['lyrics']
        song_title = elm['title']
        lyrics.append(lyric)
        lyric_titles.append(song_title)

    return queries, query_titles, lyrics, lyric_titles

queries, query_titles, lyrics, lyric_titles = get_dataset("queries.json", "lyrics.json")

song_dataset = CustomSongDataset(lyrics, lyric_titles , tokenizer, 512)
query_dataset = CustomSongDataset(queries, query_titles , tokenizer, 512)

song_data_loader = torch.utils.data.DataLoader(song_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0
)

query_data_loader = torch.utils.data.DataLoader(query_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0
)

query_embeddings = []
query_titles = []
query_bert_embeddings = []

for batch_idx, data in enumerate(query_data_loader):
    ids = data['input_ids'].to(device, dtype = torch.long)
    mask = data['attention_mask'].to(device, dtype = torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
    titles = data['titles']
    outputs = model(ids, mask, token_type_ids)
    # query_titles.append(titles)
    # query_embeddings.append(outputs[0])
    print(list(outputs[0].detach().cpu().numpy()))
    query_bert_embeddings.append([titles, outputs[0][0].detach().cpu().numpy()])
np.save("query_bert_embeddings",query_bert_embeddings)



song_embeddings = []
song_titles = []
song_bert_embeddings = []
for batch_idx, data in enumerate(song_data_loader):
    ids = data['input_ids'].to(device, dtype = torch.long)
    mask = data['attention_mask'].to(device, dtype = torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
    titles = data['titles']
    outputs = model(ids, mask, token_type_ids)
    # song_titles.append(titles)
    # song_embeddings.append(outputs[0])
    print(list(outputs[0].detach().cpu().numpy()))
    song_bert_embeddings.append([titles, outputs[0].detach().cpu().numpy()])

np.save("song_bert_embeddings",query_bert_embeddings)



