# fake_newz_prediction
Fake news detection identifies and classifies misleading or false information using machine learning models trained on real and fake news datasets.

Fake News Prediction - Overview  

Steps in Fake News Prediction   
1. Data Collection – Load a dataset containing news articles and their labels (real or fake).  
2. Preprocessing – Convert text data into numerical format using TF-IDF Vectorization.  
3. Model Training – Train a Naïve Bayes classifier on the processed data.  
4. Model Evaluation – Split the data into training & testing sets and calculate accuracy.  
5. Deployment with Flask API – Create an API to predict fake news from user input.  

---

Features of Fake News Prediction System 
- Dataset: Contains news text and corresponding labels (real or fake).  
- Text Processing: Uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.  
- Classification Model: Multinomial Naïve Bayes, trained for text classification.  
- Web API: Flask-based API to serve predictions.  

---

Technologies Used  
- Python – Main programming language.  
- Pandas & NumPy – Data handling & manipulation.  
- Scikit-Learn – Machine learning library for model training.  
- Flask – Web framework for API deployment.  
- Joblib/Pickle (.pkl files) – To save and load trained models efficiently.  

---

*What is .pkl (Pickle) File?
A *pkl* (Pickle) file stores trained machine learning models and vectorizers in a serialized format, allowing them to be loaded later without retraining

# train_model.py
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset here load your csv file i fetched it fron kaggle then specify your doc path
data = pd.read_csv("C:/Users/Documents/fake newz/data.csv")  # Ensure this file exists

# Print column names to verify
print("Column Names:", data.columns)

# Ensure correct column names (update if needed)
if 'text_column' in data.columns and 'label_column' in data.columns:
    data.rename(columns={'text_column': 'text', 'label_column': 'label'}, inplace=True)

# Check if required columns exist
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("Dataset does not contain required columns: 'text' and 'label'.")

# Replace numeric labels with string labels
data['label'] = data['label'].map({1: 'fake news', 0: 'real news'})

# Preprocess the data (vectorization)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']  # Updated with the new label format

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate model accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Print sample data for verification
print(data.head())
print(data.dtypes)

#app.py

from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Check if model files exist before loading
model_path = "fake_news_model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found! Train the model first.")

# Load Model & Vectorizer
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("text")
    if not data:
        return jsonify({"error": "No text provided"}), 400

    transformed_text = vectorizer.transform([data])
    prediction = model.predict(transformed_text)

    return jsonify({"Fake News Prediction": bool(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)


