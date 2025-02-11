import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import string
import nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# 确保下载了必要的nltk资源
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Text Cleaning: 去除标点符号
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)

    # Lowercasing
    text_lower = text_no_punct.lower()

    # Tokenization
    tokens = word_tokenize(text_lower)

    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming or Lemmatization (这里使用Lemmatization)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text

# Load data
data = pd.read_csv('business_data.csv')

# Data preprocessing
#data['content'] = data['content'].apply()
data['content'] = data['content'].apply(preprocess_text)

# Split data into training and testing sets
X = data['content']
y = data['category']
label_counts = y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

from sklearn.naive_bayes import MultinomialNB
# Example with Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(max_iter=1000)
#model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")