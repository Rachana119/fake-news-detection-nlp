# -------------------------------
# Fake News Detection - Model Training
# -------------------------------

import pandas as pd
import numpy as np
import re
import pickle

# NLP
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# 1. Load Dataset
# -------------------------------

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = 0   # Fake
true_df["label"] = 1   # Real

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Label distribution:")
print(df["label"].value_counts())


# -------------------------------
# 2. Text Preprocessing (NLP)
# -------------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("\nSample cleaned text:")
print(df[["text", "clean_text"]].head())


# -------------------------------
# 3. Feature Extraction (TF-IDF)
# -------------------------------

X = df["clean_text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF shape:", X_tfidf.shape)


# -------------------------------
# 4. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# 5. Train Machine Learning Model
# -------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# -------------------------------
# 6. Model Evaluation
# -------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------
# 7. Save Model & Vectorizer
# -------------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
