import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('business_data.csv')
texts = data['content'].values
labels = data['category'].values

label_mapping = {'business': 0, 'tradee': 1}
labels = np.array([label_mapping[label] for label in labels])

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(len(seq) for seq in X_train_seq)
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_length)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_length)

inputs = tf.keras.Input(shape=(max_length,))
embedding_layer = tf.keras.layers.Embedding(input_dim=5000, output_dim=128)(inputs)
attention = tf.keras.layers.Attention()([embedding_layer, embedding_layer])
flatten = tf.keras.layers.Flatten()(attention)
outputs = tf.keras.layers.Dense(2, activation='softmax')(flatten)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_split=0.1)

y_pred = np.argmax(model.predict(X_test_padded), axis=-1)

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['business', 'tradee']))
