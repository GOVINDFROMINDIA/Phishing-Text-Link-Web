import streamlit as st
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

# Dictionary to store learned links
learned_links = {"dangerous": set(), "safe": set()}

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
                prediction = predict_link_type(link)
                st.write(f"Prediction for {link}: {prediction}")

                if prediction == 'benign':
                    if st.button(f"Report {link} as Dangerous"):
                        learned_links["dangerous"].add(link)
                        st.write(f"Link {link} learned as dangerous")
                else:
                    if st.button(f"Report {link} as Safe"):
                        learned_links["safe"].add(link)
                        st.write(f"Link {link} learned as safe")

    # Display learned links
    if learned_links["dangerous"]:
        st.subheader("Learned Dangerous Links:")
        st.write(learned_links["dangerous"])
    if learned_links["safe"]:
        st.subheader("Learned Safe Links:")
        st.write(learned_links["safe"])

if __name__ == "__main__":
    main()
