import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import re
nltk.download('stopwords')
nltk.download('punkt', download_dir='C:\\Users\\entropy\\nltk_data')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

train_df = pd.read_csv('twitter_training.csv', header=None)
train_df.columns = ['id', 'product_name', 'sentiment', 'review']
train_df['review'] = train_df['review'].fillna('')

validation_df = pd.read_csv('twitter_validation.csv', header=None)
validation_df.columns = ['id', 'product_name', 'sentiment', 'review']

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'@\w+|#\w+', '', text)
        tokens = word_tokenize(text)
        return tokens
    else:
        return []

train_df['cleaned_review'] = train_df['review'].apply(preprocess_text)
validation_df['cleaned_review'] = validation_df['review'].apply(preprocess_text)

train_df['cleaned_review'] = train_df['cleaned_review'].apply(lambda x: ' '.join(x))
validation_df['cleaned_review'] = validation_df['cleaned_review'].apply(lambda x: ' '.join(x))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__min_df': [1, 3, 5],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.01, 0.1, 1]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(train_df['cleaned_review'], train_df['sentiment'])

print("Best Parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(validation_df['cleaned_review'])

accuracy = accuracy_score(validation_df['sentiment'], y_pred)
print(f'Accuracy with best hyperparameters: {accuracy:.2f}')

results_df = pd.DataFrame({
    'review': validation_df['review'],
    'actual': validation_df['sentiment'],
    'predicted': y_pred
})

misclassified = results_df[results_df['actual'] != results_df['predicted']]
print(misclassified)

joblib.dump(best_model, 'sentiment_model.pkl')
joblib.dump(grid_search.best_estimator_.named_steps['tfidf'], 'tfidf_vectorizer.pkl')
