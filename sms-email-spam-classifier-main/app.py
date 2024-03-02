import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# provide actual path of file 
current_directory = os.path.dirname(os.path.abspath(__file__))
pickle_file_path = os.path.join(current_directory, 'vectorizer.pkl')
tfidf = pickle.load(open(pickle_file_path, 'rb'))
     #tfidf = pickle.load(open('vectorizer.pkl','rb'))
current_directory = os.path.dirname(os.path.abspath(__file__))
pickle_file_path = os.path.join(current_directory, 'model.pkl')
model = pickle.load(open(pickle_file_path, 'rb'))
   #model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
