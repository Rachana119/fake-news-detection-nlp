import streamlit as st
import pickle
import re
import numpy as np

# -------------------------------
# Load trained model & vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Text preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write(
    "This system uses **NLP and Machine Learning** to classify news as "
    "**Fake or Real**, and explains the prediction using influential words."
)

st.info(
    "📌 Best results are obtained with a **headline + full news article** "
    "(minimum 30 words)."
)

# -------------------------------
# User Inputs
# -------------------------------
headline = st.text_input(
    "News Headline",
    placeholder="Enter the news headline here..."
)

article = st.text_area(
    "News Article",
    height=220,
    placeholder="Paste the full news article here (minimum 30 words)..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Check News"):
    if headline.strip() == "" or article.strip() == "":
        st.warning("⚠️ Please enter both headline and article.")
    
    elif len(article.split()) < 30:
        st.warning(
            "⚠️ Please enter a **full news article (at least 30 words)** "
            "for accurate prediction."
        )
    
    else:
        # Combine headline + article
        combined_text = headline + " " + article

        # Clean and vectorize
        cleaned_text = clean_text(combined_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        # Show prediction result
        if prediction == 0:
            st.error("❌ Fake News Detected")
            confidence = probability[0]
        else:
            st.success("✅ Real News Detected")
            confidence = probability[1]

        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        # -------------------------------
        # Explainable AI (Top Influential Words)
        # -------------------------------
        st.subheader("🔍 Why this prediction?")

        feature_names = np.array(vectorizer.get_feature_names_out())
        coefficients = model.coef_[0]

        # Get non-zero TF-IDF indices
        non_zero_indices = vectorized_text.nonzero()[1]

        # Get word contributions
        word_contributions = coefficients[non_zero_indices]
        words = feature_names[non_zero_indices]

        # Combine and sort by absolute influence
        word_importance = sorted(
            zip(words, word_contributions),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Show top 10 influential words
        st.write("Top words influencing the decision:")
        for word, score in word_importance[:10]:
            direction = "REAL" if score > 0 else "FAKE"
            st.write(f"• **{word}** → pushes towards **{direction}**")
