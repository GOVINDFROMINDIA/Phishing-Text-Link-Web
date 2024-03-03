import os
import streamlit as st
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sqlite3


connection = sqlite3.connect("database.db")

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained models
model = joblib.load('voting_classifier.joblib')
with open('check_spam_classifier.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)
with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load labels
with open('labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define directory and file paths
feedback_dir = "feedback"
danger_file = os.path.join(feedback_dir, "dangerous_links.txt")
safe_file = os.path.join(feedback_dir, "safe_links.txt")

# Create directory if it doesn't exist
os.makedirs(feedback_dir, exist_ok=True)

def split_text_and_links(text):
    text_content = ""
    link_content = ""
    for word in text.split():
        if word.startswith("http") or word.startswith("www"):
            link_content += word + "\n"
        else:
            text_content += word + " "
    return text_content.strip(), link_content.strip()

def predict_link_type(link):
    new_data = [link]
    prediction = model.predict(new_data)[0]
    return prediction

def predict_scam(text):
    text = preprocess_input(text)
    text_tfidf = vectorizer.transform([text])
    prediction = clf.predict(text_tfidf)[0]
    return labels[prediction]

def preprocess_input(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def report_link(file, link):
    with open(file, 'a') as f:
        # f.write(link + '\n')
        connection.execute(f"""insert into links values("{link}" , {1})""")

def is_reported(file, link):
    # if os.path.exists(file):
        # with open(file, 'r') as f:
            # return link.strip() in f.readlines()
    data = connection.execute("""select * from links where link = "{link}" """)
    if len(list(data)) > 0:
        return True

    return False

def main():
    st.title("Scam Predictor")

    input_text = st.text_area("Enter your text here:")

    if st.button("Predict"):
        text_content, link_content = split_text_and_links(input_text)

        if text_content:
            st.subheader("Text:")
            st.write(text_content)

            prediction = predict_scam(text_content)
            st.write(f"Prediction: {prediction}")

        if link_content:
            st.subheader("Links:")
            st.write(link_content)

            for link in link_content.splitlines():
                if is_reported(danger_file, link):
                    st.write(f"Link {link} learned as dangerous")
                elif is_reported(safe_file, link):
                    st.write(f"Link {link} learned as safe")
                else:
                    prediction = predict_link_type(link)
                    st.write(f"Prediction for {link}: {prediction}")

                    if prediction == 'benign':
                        if st.button(f"Report {link} as Dangerous"):
                            report_link(danger_file, link)
                            st.write(f"Link {link} learned as dangerous")
                    else:
                        if st.button(f"Report {link} as Safe"):
                            report_link(safe_file, link)
                            st.write(f"Link {link} learned as safe")

if __name__ == "__main__":
    main()
