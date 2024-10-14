# Sentiment Analysis Tool - Model Training

This project is a sentiment analysis tool that uses a Naive Bayes classifier to classify text (e.g., product reviews, tweets) as positive, negative, or neutral. The tool processes text data, extracts features using the TF-IDF method, and trains a machine learning model using the Naive Bayes algorithm. It also includes functionality for hyperparameter tuning with GridSearchCV and saves the trained model for future use.

## Features

- Preprocessing of text data, including tokenization and stopword removal.
- Use of TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to transform text into numerical features.
- Naive Bayes classifier for sentiment classification.
- Hyperparameter tuning using GridSearchCV for model optimization.
- Accuracy evaluation and misclassification identification.
- Saving the trained model and vectorizer for reuse.

## Prerequisites

Before running the code, ensure you have the following packages installed:

- `pandas`
- `nltk`
- `scikit-learn`
- `joblib`

You can install the required packages using the following:

```bash
pip install pandas nltk scikit-learn joblib
