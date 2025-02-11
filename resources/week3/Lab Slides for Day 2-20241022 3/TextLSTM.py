import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

file_path = 'business_data.csv'
data = pd.read_csv(file_path)
texts = data['content'].tolist()
categories = data['category'].tolist()

vocab_size = 10000
embedding_dim = 128
maxlen = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(categories)

x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

predictions = model.predict(x_test)
predicted_classes = (predictions > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, predicted_classes)

print("Confusion Matrix:")

print(cm)
from sklearn.metrics import classification_report
class_names = label_encoder.classes_
report = classification_report(y_test, predicted_classes, target_names=class_names, output_dict=True)
for key, value in report.items():
    print(f"{key}: {value}")