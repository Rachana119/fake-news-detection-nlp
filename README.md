# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Project Overview
This project is an **NLP-based Fake News Detection system** that classifies news articles as **Fake** or **Real** using **Machine Learning techniques**. The system analyzes both the **headline and full article text**, extracts meaningful linguistic features, and predicts the authenticity of news content.  
To enhance transparency, the project also includes **Explainable AI**, highlighting key words that influenced the model’s decision.

The application is deployed as an **interactive web application using Streamlit**.

---

## 🎯 Objectives
- Detect fake news using **Natural Language Processing (NLP)**
- Apply **Machine Learning** for text classification
- Improve prediction accuracy using **headline + article context**
- Provide **Explainable AI** for model interpretability
- Deploy a user-friendly **Streamlit web application**

---

## 🧠 How the Project Works
1. User enters a **news headline** and **full article**
2. Text is **cleaned and preprocessed**
3. Text is converted into numerical features using **TF-IDF**
4. A trained **Logistic Regression model** predicts Fake or Real
5. **Explainable AI** displays influential words affecting prediction
6. Final result and confidence score are shown to the user

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **NLP:** Text preprocessing, TF-IDF  
- **Machine Learning:** Logistic Regression  
- **Explainable AI:** Feature weight analysis  
- **Web Framework:** Streamlit  
- **Libraries:** pandas, numpy, scikit-learn, nltk  


---

## 📂 Project Structure

fake-news-detection/
│
├── app.py                
├── train_model.py        
├── model.pkl            
├── vectorizer.pkl         
├── requirements.txt       
├── README.md              
└── data/                  

## 📊 Dataset
- **ISOT Fake News Dataset**
- Contains labeled **Real** and **Fake** news articles
- Source: Kaggle  
- Dataset files are not included in this repository due to size limitations

---

## 🚀 Features
- Headline + article-based prediction
- Minimum-length validation for reliable results
- Confidence score display
- Explainable AI (top influential words)
- Clean and interactive Streamlit interface

---



---

