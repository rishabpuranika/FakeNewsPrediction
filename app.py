# Import necessary libraries
import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load data
try:
    news_df = pd.read_csv('C:/Users/risha/Downloads/4th Sem/4th Sem/Miniproject/train.csv') #Filepath of train.csv
except FileNotFoundError:
    st.error("The file 'train.csv' was not found.")
    st.stop()

# Fill missing values
news_df = news_df.fillna('')

# Combine author and title into a single content column
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Streamlit app
st.title('Fake News Detector')
#Entering the news article title which has been stemmed using sentence.py
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')
