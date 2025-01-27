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
A *pkl* (Pickle) file stores trained machine learning models and vectorizers in a serialized format, allowing them to be loaded later without retrainin
