import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, pad_idx, num_classes):
        super(TextCNN, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(
                ks, embedding_dim)) for ks in kernel_sizes]
        )   # conv layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=channels *
                            len(kernel_sizes), out_features=num_classes)

    def forward(self, text):
        embedded = self.embedding(text)  # [B, L, E]
        embedded = embedded.unsqueeze(1)    # [B, 1, L, E]
        conved = [F.relu(conv(embedded)).squeeze(3)
                  for conv in self.convs]   # [B, C, L] * len(kernel_sizes)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]  # [B, C] * len(kernel_sizes)
        cat = torch.cat(pooled, dim=1)  # [B, C * len(kernel_sizes)]

        return self.fc(self.dropout(cat))   # [B, num_classes]


class TextCNNModel:
    def __init__(self, vocab_size, embedding_dim, channels, kernel_sizes, pad_idx, num_classes):
        self.model = TextCNN(vocab_size, embedding_dim,
                             channels, kernel_sizes, pad_idx, num_classes)
        self.criterion = nn.CrossEntropyLoss()  # multi_class classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, dataloader, epochs=10):
        self.model.train()
        # training loop
        for epoch in range(epochs):
            losses = []
            for X, y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                losses = [loss.item()]
                loss.backward()  # compute gradients
                self.optimizer.step()
                # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(losses)}")

    def predict(self, testloader):
        self.model.eval()
        y_all, y_pred_all = [], []
        with torch.no_grad():
            # testing loop
            for X, y in testloader:
                outputs = self.model(X)
                # for multi_calss
                _, predicted = torch.max(outputs, 1)
                y_all.append(y)
                y_pred_all.append(predicted)
        y_all = torch.cat(y_all).numpy()
        y_pred_all = torch.cat(y_pred_all).numpy()
        return y_all, y_pred_all


def yield_tokens(texts):
    for text in texts:
        yield text.split()


if __name__ == "__main__":
    dataset = ["movie", "news", "sms"]  # datasets
    for d in dataset:
        texts, labels_i = ReadData(f"dataset/sms.csv")
        texts = texts.apply(ProcessText)

        # transform texts to vocab
        vocab = build_vocab_from_iterator(
            yield_tokens(texts), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        texts_vocab = []
        for text in texts:
            texts_vocab.append(vocab(text.split()))

        # pad to same length
        texts_to_be_padded = [torch.tensor(t) for t in texts_vocab]
        padded_texts = pad_sequence(
            texts_to_be_padded, batch_first=True, padding_value=vocab["<pad>"])
        print(f"using TextCNN, processing dataset {d}")
        # print(padded_texts.max(), padded_texts.min())
        # print(padded_texts.shape)
        # transform to pytorch dataset
        dataset = TensorDataset(
            padded_texts, torch.tensor(labels_i, dtype=torch.long))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = random_split(
            dataset, [train_size, test_size])

        batch_size = 32
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        # define TextCNN model
        model = TextCNNModel(len(vocab), 128, 100, [
            3, 4, 5], vocab["<pad>"], labels_i.max() + 1)
        model.fit(train_loader, epochs=10)

        # predict
        y_all, y_pred_all = model.predict(test_loader)
        accuracy = accuracy_score(y_all, y_pred_all)
        classification_rep = classification_report(y_all, y_pred_all)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{classification_rep}")
