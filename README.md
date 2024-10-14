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
```

Additionally, the NLTK data for stopwords and tokenization is required. This will be downloaded automatically when the script runs,
but you can also download it manually by running the following in a Python shell:

```bash
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
## Training Data
The code expects two CSV files for training and validation data:

- `twitter_training.csv`: Contains the training data.
- `twitter_validation.csv`: Contains the validation data. Both CSV files should have the following columns:
  - `id`: A unique identifier for each review.
  - `product_name`: The name of the product being reviewed.
  - `sentiment`: The sentiment of the review (e.g., positive, negative, neutral).
  - `review`: The text of the review.
## Running the Code
Download or prepare your dataset (twitter_training.csv and twitter_validation.csv).
Place the dataset files in the same directory as the code.
Run the Python script to start training the model:
```bash
python sentiment_analysis_training.py
```
### Code Functionality
- The script preprocesses the text data by:
  - Lowercasing the text.
  - Removing mentions and hashtags.
  - Tokenizing the text.
- The cleaned data is transformed using a `TfidfVectorizer`.
- A MultinomialNB (Naive Bayes) classifier is used to fit the model.
- Hyperparameter tuning is performed using GridSearchCV to optimize:
  - Maximum document frequency (max_df).
  - Minimum document frequency (min_df).
  - N-gram range (unigrams and bigrams).
  - Smoothing parameter (alpha) for Naive Bayes.
- The best model is saved as sentiment_model.pkl and the TF-IDF vectorizer as tfidf_vectorizer.pkl.
  
## Output
The best hyperparameters found during tuning are printed to the console.
The accuracy of the model on the validation dataset is displayed.
A list of misclassified reviews is printed for further analysis.

## Model and Vectorizer
After running the code, the following files will be generated:

- `sentiment_model.pkl`: The trained Naive Bayes model.
- `tfidf_vectorizer.pkl`: The TF-IDF vectorizer used to transform the input text data.

These files can be loaded in a separate script for predicting the sentiment of new reviews.

## Support
If you found this project helpful and would like to support me, you can [Buy Me a Coffee](https://buymeacoffee.com/caffeinatedkim)

## License
This project is open-source and available under the [GNU General Public License](https://github.com/kim2235/basic_sentiment_analysis/blob/main/LICENSE)

