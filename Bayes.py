from utils import *


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.class_table = None  # class table
        self.class_prior = None  # class prior
        self.feature_prob = None  # feature probability
        self.alpha = alpha  # smoothing parameter

    def fit(self, X, y):
        # get class num
        self.class_table = np.unique(y)  # already sorted
        num_classes = len(self.class_table)
        # get feature num
        data_num, num_features = X.shape

        # compute class prior log probability
        class_num = np.bincount(y)
        self.class_prior = np.log(class_num / data_num)

        # compute feature prior probability
        feature_count = np.zeros((num_classes, num_features))
        for i, c in enumerate(self.class_table):
            feature_count[i] = X[y == c].sum(axis=0)  # sum each feature

        # compute log probability
        s_feature_count = feature_count + self.alpha
        s_class_count = s_feature_count.sum(axis=1).reshape(-1, 1)
        self.feature_prob = np.log(s_feature_count) - np.log(s_class_count)

    def predict(self, X):
        # joint log likelihood, product to sum
        joint_log_likelihood = np.dot(
            self.feature_prob, X.T
        ) + self.class_prior.reshape(-1, 1)
        max_idx = np.argmax(joint_log_likelihood, axis=0)
        y_pred = self.class_table[max_idx]
        return y_pred


if __name__ == "__main__":
    dataset = ["movie", "news", "sms"]  # datasets
    for d in dataset:
        texts, labels_i = ReadData(f"dataset/{d}.csv")

        T_train, T_test, y_train, y_test = train_test_split(
            texts, labels_i, test_size=0.3, random_state=42
        )

        tker = Tokenizer()
        X_train = tker.fit_transform(T_train)
        X_test = tker.transform(T_test)
        print(f"using bayes algorithm, processing dataset {d}")
        print("tokenization finished")
        print(X_train.shape)
        bayes_model = NaiveBayes()
        bayes_model.fit(X_train, y_train)

        y_pred = bayes_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{classification_rep}")
