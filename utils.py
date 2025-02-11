import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')


def ReadData(path):
    """ read data from csv file

    Args:
        path (str): file path

    Returns:
        texts: dataframe, texts
        labels_i: array, labels
    """
    data = pd.read_csv(path)
    fields = data.columns   # get the column names
    texts = data[fields[0]]
    labels = data[fields[1]]

    # transform labels from string to int
    labels_name = labels.unique()
    l2i = {l: i for i, l in enumerate(labels_name)}
    # transform labels to int
    labels_i = np.array([l2i[l] for l in labels], dtype=int)

    return texts, labels_i


def ProcessText(text, returnStr=True):
    """ process text, remove punctuation, stop words, lemmatize

    """
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)  # remove punctuation
    text_lower = text_no_punct.lower()
    tokens = word_tokenize(text_lower)  # word tokenization
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in
                       stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(
        word) for word in filtered_tokens]  # lemmatization
    if returnStr:
        return ' '.join(lemmatized_tokens)
    else:
        return lemmatized_tokens


class Tokenizer:
    """ Tokenizer class
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)

    def fit_transform(self, texts):
        # pandas series use apply to process each element
        texts = texts.apply(ProcessText)
        X = self.vectorizer.fit_transform(texts)    # transform texts to tf-idf
        return X.toarray()

    def transform(self, texts):
        texts = texts.apply(ProcessText)
        texts = self.vectorizer.transform(texts)
        # tranform sparse matrix to array
        return texts.toarray()
