#I import all necessary libraries for the IP
import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

#Create Flask API
app = Flask(__name__)

#First I load Logistic Regression Model
with open('BBC News Summary/logisticregression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

#Then I load TfidfVectorizer
with open('BBC News Summary/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

tokenizer = DistilBertTokenizer.from_pretrained('BBC News Summary') #Now I must load BERT model and tokenizer
bert_model = DistilBertForSequenceClassification.from_pretrained('BBC News Summary', num_labels=5)
bert_model.eval()  #Its a must to set the model to evaluation mode

label_map = {0: 'business', 1: 'entertainment', 2: 'politics', 3: 'sport', 4: 'tech'} #I define a label mapping 

stop_words = set(stopwords.words('english')) #If using Logistic Regression, I need to load stopwords for preprocessing

def preprocess_text(text): #As before, here its the preprocessing function
    text = text.lower() #Convert text to lowercase
    text = re.sub(r'\d+', '', text) #Remove digits
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  #Get JSON data from the request
    text_input = data['text']  #Extract the text input from the JSON

    # Predict using Logistic Regression
    if data['model'] == 'logistic_regression':
        # Transform the cleaned text using TfidfVectorizer
        cleaned_text = preprocess_text(text_input) #Apply preprocess function
        cleaned_text = re.sub(r'\W', ' ', cleaned_text) #Remove punctuation and special characters  
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words]) #Remove stopwords
        X_tfidf = tfidf_vectorizer.transform([cleaned_text]) #Apply the TF-IDF vectorizer
        probabilities = logistic_model.predict_proba(X_tfidf)[0] #I get the probability of each class
        max_prob = max(probabilities) #I get the highest probability
        #Now I check if max probability is below 0.6, in that case it returns "Other"
        if max_prob < 0.6:
            return jsonify({'prediction': 'Other'})
        
        prediction = logistic_model.predict(X_tfidf)  #Now it makes predictions
        predicted_label = label_map[int(prediction[0])] #Map to get the label
        return jsonify({'prediction': predicted_label})  #Finally return prediction

    # Predict using BERT
    elif data['model'] == 'bert':
        cleaned_text = preprocess_text(text_input) #Apply preprocess function
        inputs = tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=512) #Tokenize the input text
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Move the inputs to the appropriate device
        bert_model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs) #Now it makes predictions
            logits = outputs.logits #Get the raw output scores, logits, as they are extracted from the outputs
        probabilities = torch.nn.functional.softmax(logits, dim=1) #Convert logits to probabilities
        max_prob = torch.max(probabilities).item()  #I get the highest probability
        #Now I check if max probability is below 0.6, in that case it returns "Other"
        if max_prob < 0.6:
            return jsonify({'prediction': 'Other'})
        
        prediction = torch.argmax(logits, dim=1).cpu().numpy()
        predicted_label = label_map[int(prediction[0])] #Map to get the label
        return jsonify({'prediction': predicted_label})  #Finally return prediction

    else:
        return jsonify({'error': 'Model not supported!'}), 400

if __name__ == '__main__':
    app.run(debug=True)
