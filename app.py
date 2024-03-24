from flask import Flask, render_template, request, redirect, url_for
import re
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK data (you only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer for link prediction
link_model = joblib.load('voting_classifier.joblib')

# Load the new scam text predictor model and vectorizer
with open('check_spam_classifier.pkl', 'rb') as clf_file:
    scam_classifier = pickle.load(clf_file)

with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    scam_vectorizer = pickle.load(vectorizer_file)

# Load labels from the text file for scam text prediction
with open('labels.txt', 'r') as labels_file:
    scam_labels = labels_file.read().splitlines()

# Define stopwords and lemmatizer for scam text prediction
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize dictionary to store reported links and their statuses
reported_links = {}

def preprocess_input(text):
    # Preprocess the input text in the same way as the training data for scam text prediction
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def predict_link_type(link):
    # Preprocess the link
    link = preprocess_input(link)
    
    # Predict link type
    link_prediction = link_model.predict([link])[0]
    
    return link_prediction

def predict_scam_text(input_text):
    # Preprocess the input text
    input_text = preprocess_input(input_text)
    
    # Vectorize the preprocessed text
    input_text_tfidf = scam_vectorizer.transform([input_text])
    
    # Make a prediction
    prediction = scam_classifier.predict(input_text_tfidf)
    
    # Get the label using the labels list
    predicted_label = scam_labels[prediction[0]]
    
    return predicted_label

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        text = request.form['text']

        # Preprocess the input text
        text = preprocess_input(text)
        
        # Split text into links and texts
        links = re.findall(r'(https?://\S+)', text)
        text_without_links = re.sub(r'(https?://\S+)', '', text)
        
        # Predict link types
        link_predictions = [predict_link_type(link) for link in links]
        
        # Predict scam text
        scam_text_prediction = predict_scam_text(text_without_links)
        
        return render_template('result.html', text=text, links=zip(links, link_predictions), scam_text=scam_text_prediction)

@app.route('/report', methods=['POST'])
def report():
    if request.method == 'POST':
        link = request.form['link']
        report_type = request.form['report_type']
        
        if link in reported_links:
            return redirect(url_for('home', message="The link has been previously reported"))
        
        reported_links[link] = report_type
        
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
