import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        num_samples, num_features = X.shape
        self.num_classes = len(np.unique(y))    # number of classes

        # init weights and bias
        self.weights = np.zeros((num_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))

        # gradient descent
        for _ in range(self.num_iterations):
            # forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)    # predicted probability

            # backward pass
            dw = (1 / num_samples) * np.dot(X.T,
                                            (y_pred - np.eye(self.num_classes)[y]))
            db = (1 / num_samples) * np.sum(y_pred -
                                            np.eye(self.num_classes)[y], axis=0, keepdims=True)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = self.scaler.transform(X)
        z = np.dot(X, self.weights) + self.bias   # linear regression
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1)


if __name__ == '__main__':
    dataset = ["movie", "news", "sms"]  # datasets
    for d in dataset:
        texts, labels_i = ReadData(f"dataset/{d}.csv")

        T_train, T_test, y_train, y_test = train_test_split(
            texts, labels_i, test_size=0.3, random_state=42)

        tker = Tokenizer()
        X_train = tker.fit_transform(T_train)
        X_test = tker.transform(T_test)
        print(f"using logistic regression algorithm, processing dataset {d}")
        print("tokenization finished")
        print(X_train.shape)
        LR_model = LogisticRegression()
        LR_model.fit(X_train, y_train)

        y_pred = LR_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{classification_rep}")
