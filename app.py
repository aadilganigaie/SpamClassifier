import streamlit as st
import pickle
import re
import string
import skleran
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open('mnbmodel.pickle', 'rb'))
tfidf = pickle.load(open('vectorizer.pickle', 'rb'))
preprocess = pickle.load(open('preprocess.pickle', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = preprocess(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
