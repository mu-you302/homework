import os
import codecs
import re
from nltk.stem import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfTokenizer:
    def __init__(self) -> None:
        self.vocab_in_doc_num = {}
        self.vocab_table = {}
        self.num_docs = 0
        self.dims = 0

    def update_vocab(self, docs):
        for doc in docs:
            self.num_docs += 1
            words = set(doc)
            for w in words:
                self.vocab_in_doc_num[w] = self.vocab_in_doc_num.get(w, 0) + 1
        self.vocab_table = {w: i for i, w in enumerate(
            self.vocab_in_doc_num.keys())}
        self.dims = len(self.vocab_in_doc_num)

    def _idfs(self):
        doc_freq = np.array(list(self.vocab_in_doc_num.values()))

        # return np.log(self.num_docs / (doc_freq + 1.))
        return np.log(self.num_docs / (doc_freq))

    def fit(self, docs):
        self.update_vocab(docs)
        idfs = self._idfs().reshape(1, -1)
        word_freq = np.zeros((self.num_docs, self.dims))

        for i, doc in enumerate(docs):
            for word in doc:
                word_freq[i, self.vocab_table[word]] += 1
        tf = word_freq / word_freq.sum(axis=1, keepdims=True)

        tf_idf = np.log(tf + 1) * idfs

        tf_idf = tf_idf / np.sum(tf_idf**2, axis=1, keepdims=True)**0.5
        print(np.sum(tf_idf == 0)/tf_idf.size)
        print(self.dims)

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
