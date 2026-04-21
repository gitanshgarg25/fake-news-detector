# Fake News Detection using Machine Learning and NLP

A machine learning project that detects whether a news article is real or fake using Natural Language Processing techniques. This project applies text preprocessing, TF-IDF vectorization, and a Passive Aggressive Classifier to classify news content with high accuracy. It also includes a simple prediction interface for testing custom news text.

## Project Overview

The rapid spread of misinformation on digital platforms has made fake news detection an important real-world problem. This project aims to classify news articles as real or fake by learning patterns from labeled textual data. The model is trained on a dataset containing article text and corresponding labels, then used to predict the authenticity of unseen news content.

## Problem Statement

The objective of this project is to build a machine learning system that can identify fake news articles from textual content. By analyzing writing patterns and word usage, the model helps distinguish unreliable news from reliable information.

## Features

- Detects fake and real news from article text
- Uses NLP-based text preprocessing
- Applies TF-IDF vectorization for feature extraction
- Trains a Passive Aggressive Classifier for binary classification
- Saves trained model and vectorizer for reuse
- Supports custom input prediction
- Includes an interactive interface for testing predictions

## Dataset

The project uses a labeled fake news dataset containing article-related fields such as:

- `id`
- `title`
- `author`
- `text`
- `label`

In the notebook workflow, unnecessary columns are removed and the main model is trained using the news article `text` field and its corresponding `label`.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Pickle
- Gradio / Flask-based deployment concept
- Jupyter Notebook

## Methodology

### 1. Data Loading
The dataset is loaded from CSV files and inspected for structure, null values, and label distribution.

### 2. Data Cleaning
The preprocessing pipeline includes:
- removing unwanted columns
- handling missing values
- converting text to lowercase
- removing non-alphabetic characters
- tokenization
- stopword removal
- lemmatization

### 3. Train-Test Split
The text data is split into training and testing sets to evaluate model performance on unseen data.

### 4. Feature Extraction
TF-IDF vectorization is used to convert textual data into numerical form so that it can be used by the machine learning model.

### 5. Model Training
A Passive Aggressive Classifier is trained on the transformed text data for fake news classification.

### 6. Evaluation
The model is evaluated using classification metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Performance

The project report shows that the Passive Aggressive Classifier performed strongly for this task, with reported accuracy around **94% to 96%**, making it effective for distinguishing fake and real news articles. :contentReference[oaicite:1]{index=1}

## Project Structure

```bash
Fake-News-Detection/
│── newsdetection.ipynb
│── app.py
│── model.pkl
│── vector.pkl
│── requirements.txt
│── README.md
│── train.csv
│── test.csv
│── images/
