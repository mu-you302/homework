import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from nltk.stem import SnowballStemmer
from collections import deque

stemmer = SnowballStemmer("english")


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


def stemText(text):
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(t) for t in text]  # stem the text

    return "".join(text)


def tokenize(texts):
    """ tokenize texts

    Args:
        texts: dataframe

    Returns:
        X: tokenized texts by tf-idf
    """
    # texts = texts.apply(stemText)
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(texts)   # transform texts to tf-idf
    X = X.toarray()
    return X


def dist(x, y):
    """ calculate the (cos) distance between two samples
    """
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))   # cosine similarity


def findNeighbors(X, p, eps):
    """ find the neighbors of a point p

    Args:
        X : all samples
        p : one selected sample
        eps : maximum distance

    Returns:
        neighbors: all neighbors of p
    """
    neighbors = []
    for i, q in enumerate(X):
        if dist(p, q) < eps:    # if the distance between p and q is less than eps
            neighbors.append(i)
    return neighbors


def DBSCANCluster(X, eps=2, min_samples=5):
    """ cluster the data using DBSCAN

    Args:
        X : all samples
        eps (int): max distance
        min_samples (int): minimum number of samples
    """
    labels = np.zeros(X.shape[0], dtype=int)
    cluster_num = 0
    for i in range(X.shape[0]):
        if labels[i] != 0:
            continue
        neighbors = findNeighbors(X, X[i], eps)
        if len(neighbors) < min_samples:    # noise
            labels[i] = -1
        else:   # core point
            cluster_num += 1
            labels[i] = cluster_num
            # extend the cluster
            queue = deque(neighbors)    # use queue to store the neighbors
            while queue:
                neighbor = queue.popleft()
                if labels[neighbor] == -1:  # if the neighbor is noise
                    labels[neighbor] = cluster_num
                elif labels[neighbor] == 0:  # not visited
                    labels[neighbor] = cluster_num
                    # find the neighbors of the neighbor
                    new_neighbors = findNeighbors(X, X[neighbor], eps)
                    if len(new_neighbors) >= min_samples:
                        queue.extend(new_neighbors)

    return labels


def MapResults(labels, pred):
    """ map the predicted clusters to the true labels

    Args:
        labels: true labels
        pred: predicted labels
    """
    valid = (pred != -1)    # remove noise
    labels1 = labels[valid]
    pred1 = pred[valid]

    conf_matrix = confusion_matrix(labels1, pred1)  # confusion matrix
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}  # mapping
    mapped_labels = []
    for p in pred:
        if p in mapping:
            # map the predicted label to the true label
            mapped_labels.append(mapping[p])
        else:
            mapped_labels.append(-1)    # -1 means invalid label
    mapped_labels = np.array(mapped_labels)

    return mapped_labels


def ParamTuning(X, labels_i):
    """ tune the parameters of DBSCAN

    Args:
        X : all samples
        labels_i : true labels

    Returns:
        (eps, min_samples): best parameters
    """
    eps = np.arange(0.6, 1.6, 0.1)  # search range of eps
    min_samples = np.arange(5, 18, 2)
    f1s = np.zeros((len(eps), len(min_samples)))
    for i, e in enumerate(eps):
        for j, m in enumerate(min_samples):
            pred = DBSCANCluster(X, eps=e, min_samples=m)   # cluster the data
            mapped_labels = MapResults(labels_i, pred)
            if len(mapped_labels) == 0:   # no valid labels
                continue
            f1s[i, j] = f1_score(labels_i, mapped_labels, average='weighted')
    max_index_flat = np.argmax(f1s)
    # find the best parameters
    max_index = np.unravel_index(max_index_flat, f1s.shape)
    print(f"best eps: {eps[max_index[0]]}, best min_samples: {
          min_samples[max_index[1]]}")
    print(f1s.max(), f1s.argmax(), f1s)
    return eps[max_index[0]], min_samples[max_index[1]]


def ShowResults(labels, pred, dataset, alg="DBSCAN"):
    """ show the results

    Args:
        labels: true labels
        pred: predicted labels
    """
    print(f"Algorithm: {alg}, Dataset: {dataset}")
    # calculate precision, recall and f1
    precision = precision_score(labels, pred, average='weighted')
    recall = recall_score(labels, pred, average='weighted')
    f1 = f1_score(labels, pred, average='weighted')
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


dataset = ["news", "movie", "mail"]  # datasets

for d in dataset:
    texts, labels_i = ReadData(f"data/{d}.csv")

    X = tokenize(texts)

    eps, min_samples = ParamTuning(X, labels_i)  # tune the parameters

    # cluster the data
    pred = DBSCANCluster(X, eps=eps, min_samples=min_samples)

    mapped_labels = MapResults(labels_i, pred)  # map the results

    ShowResults(labels_i, mapped_labels, d)
