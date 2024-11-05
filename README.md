# FakeNewsPrediction

## Table of Contents 
1. [Fake News Detection System](#fake-news-detection-system)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Installing Required Packages](#installing-required-packages)
5. [Dataset](#dataset)
6. [Usage](#usage)
7. [Run the application](#run-the-application)
8. [Project Overview](#project-overview)
9. [License](#license)

## Fake News Detection System <a name="fake-news-detection-system"></a>
A Python-based application to detect fake news using Natural Language Processing (NLP) and machine learning techniques. This project uses logistic regression on TF-IDF vectorized text data to predict whether a given news article is real or fake. The application is built with Streamlit for a web interface and relies on scikit-learn for machine learning.

## Features <a name="features"></a>
- Data Preprocessing: Cleans, stems, and vectorizes text data.
- Fake News Detection: Predicts if the news is real or fake using a logistic regression model.
- Interactive Web Interface: Uses Streamlit for a simple and user-friendly web application.

## File Structure <a name="file-structure"></a>
- sentence.py: Contains text processing functions, including stemming of input text.
- app.py: Implements the core functionality of the fake news detection, including data processing, model training, and web interface setup using Streamlit.

## Installing Required Packages <a name="installing-required-packages"></a>
- Install the required packages by running:
```bash
pip install -r requirements.txt
```

## Dataset <a name="dataset"></a>
The model requires a labeled dataset named train.csv with the following columns:
- author: Author of the article (text).
- title: Title of the article (text).
- label: 1 if the news is fake, 0 if real (binary).
You can download the train.csv file from [Kaggle](https://www.kaggle.com/competitions/fake-news/data?select=train.csv)

## Usage <a name="usage"></a>
- Navigate into the location of the files and see to it that all the files are located in the same folder
- Select a News article Headline either from online sources or from train.csv
- You can either directly run the sentence.py file or type the following in the terminal:
```bash
python sentence.py
```
- Copy the stemmed output of the headline
- Ensure train.csv is in the same directory as app.py.

### Run the application <a name="run-the-application"></a>
```bash
streamlit run app.py
```
- Input News Article: Enter the news article's text in the input field to see if it is classified as real or fake.

## Project Overview <a name="project-overview"></a>
1. sentence.py
Processes individual text inputs:
- Prompts for user input.
- Performs stemming and removes stopwords.
2. app.py
Main application file:
- Loads and preprocesses train.csv.
- Builds a logistic regression model to classify news as real or fake.
- Provides a web interface using Streamlit, allowing users to enter article text for prediction.

## License <a name="license"></a>
This project is licensed under the MIT License .See the [LICENSE](LICENSE) file for details.