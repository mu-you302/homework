import os
import codecs
import re
from nltk.stem import SnowballStemmer
import numpy as np


class TfIdfTokenizer:
    def __init__(self) -> None:
        self.document_freq = {}  # word to document frequency
        self.vocab_table = {}   # word to idx table
        self.num_docs = 0   # number of documents
        self.dims = 0   # total word num

    def update_vocab(self, docs):
        for doc in docs:
            self.num_docs += 1
            words = set(doc)    # keep unique words
            for w in words:
                # for a word, count the num of documents that contain it
                self.document_freq[w] = self.document_freq.get(w, 0) + 1
        self.document_freq = dict(
            sorted(self.document_freq.items(), key=lambda x: x[1], reverse=True))
        self.vocab_table = {w: i for i, w in enumerate(
            self.document_freq.keys())}  # word to index table
        self.dims = len(self.document_freq)  # number of unique words

    def _idfs(self):
        # convert dict value to array
        doc_freq = np.array(list(self.document_freq.values()))
        # calculate idf
        return np.log(self.num_docs / (doc_freq))

    def fit(self, docs):
        self.update_vocab(docs)  # update documents info
        idfs = self._idfs()  # calculate idf
        word_freq = np.zeros((self.num_docs, self.dims)
                             )    # word frequency array

        for i, doc in enumerate(docs):
            for word in doc:
                word_freq[i, self.vocab_table[word]] += 1   # count
        # calculate frequency
        tf = word_freq / word_freq.sum(axis=1, keepdims=True)

        tf_idf = np.log(tf + 1) * idfs  # calculate tf-idf
        # normalize the representation
        tf_idf = tf_idf / np.sum(tf_idf**2, axis=1, keepdims=True)**0.5
        print(np.sum(tf_idf == 0)/tf_idf.size)
        print(self.dims)
        return tf_idf


class TfIdfMaxdim(TfIdfTokenizer):
    def __init__(self, max_features=5000) -> None:
        super().__init__()
        self.max_features = max_features    # max number of features
        self.term_freq = {}  # term frequency

    def update_vocab(self, docs):
        for doc in docs:
            self.num_docs += 1
            words = set(doc)    # keep unique words
            for w in words:
                # for a word, count the num of documents that contain it
                self.document_freq[w] = self.document_freq.get(w, 0) + 1
                self.term_freq[w] = self.term_freq.get(w, 0) + doc.count(w)
        # sort by term frequency and select top max_features
        self.term_freq = dict(sorted(self.term_freq.items(
        ), key=lambda x: x[1], reverse=True)[:self.max_features])
        self.document_freq = {
            k: self.document_freq[k] for k in self.term_freq.keys()}
        self.vocab_table = {w: i for i, w in enumerate(
            self.document_freq.keys())}  # word to index table
        self.dims = len(self.document_freq)  # number of unique words

    def fit(self, docs):
        self.update_vocab(docs)
        idfs = self._idfs()
        word_freq = np.zeros((self.num_docs, self.dims)
                             )    # word frequency array
        for i, doc in enumerate(docs):
            for word in doc:
                dim_idx = self.vocab_table.get(word, None)  # get index of word
                if dim_idx is not None:
                    word_freq[i, dim_idx] += 1   # count
        # calculate frequency
        tf = word_freq / word_freq.sum(axis=1, keepdims=True)

        tf_idf = np.log(tf + 1) * idfs  # calculate tf-idf
        # normalize the representation
        tf_idf = tf_idf / np.sum(tf_idf**2, axis=1, keepdims=True)**0.5

        # print(np.sum(tf_idf == 0)/tf_idf.size)
        # print(self.dims)

        return tf_idf


# load stopwords from txt file
with open(r"stopwords.txt", "r", encoding="utf-8") as fp:
    stopwords = fp.read().splitlines()
stopwords = set(stopwords)

stemmer = SnowballStemmer("english")    # Snowball stemmer

labels = os.listdir("./dataset")
l2i = {labels[i]: i for i in range(len(labels))}
texts, y = [], []

for l in labels:
    files = os.listdir(f"dataset/{l}")
    for f in files:
        path = f"dataset/{l}/{f}"
        # read txt file
        file = codecs.open(path, 'r', 'Latin1')
        content = file.read()
        # split into words
        content = content.lower()
        content = content.split()
        # remove the non-alphabet ch
        td = [re.sub(r"[^a-z]", "", w).strip() for w in content]
        td = [w for w in td if w != ""]  # remove empty string

        # stemming
        text = []
        for word in td:
            if word not in stopwords:
                text.append(stemmer.stem(word))

        texts.append(text)
        y.append(l2i[l])

tokenizer = TfIdfTokenizer()
Aik = tokenizer.fit(texts)
print(Aik)
# save to npz
np.savez("tfidf.npz", X=Aik)

tokenizer_maxdim = TfIdfMaxdim()
Aik_maxdim = tokenizer_maxdim.fit(texts)
print(Aik_maxdim)
# save to npz
np.savez("tfidf_maxdim.npz", X=Aik_maxdim)
