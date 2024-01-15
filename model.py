# libraries
import warnings
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

warnings.filterwarnings("ignore")

# reading dataset
df = pd.read_csv(r'train.csv')

# drop unnecessary column Title
df.drop(['Title'], axis=1, inplace=True)

'''
preprocesses the reviews column by converting it to lowercase and removing URLs, square brackets, non-alphabetic characters,
and stop words. It then tokenizes the text, filters out stop words, and returns the cleaned text.
'''


def preprocess_text(text):
    text = text.lower()
    text = re.sub(
        r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words).strip()


df['Review'] = df['Review'].apply(preprocess_text)

# tokenizes it into words, applies stemming (reducing words to their base form) to each word


def stem_text(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


df['Review'] = df['Review'].apply(stem_text)

# Count words from the 'Review' column
count = Counter(' '.join(df['Review']).split())

# Create a DataFrame from the word counts
words = pd.DataFrame(count.items(), columns=['Words', 'Frequency'])

# Sort by frequency and reset the index
words = words.sort_values('Frequency', ascending=False).reset_index(drop=True)

# Add a Rank column
words['Rank'] = words.index + 1
words = words[['Rank', 'Words', 'Frequency']]

# Instantiate the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(df['Review'])
y_train = df['Polarity']

# Initialize the Naive Bayes classifier
nb_clf = MultinomialNB()

# Train the Naive Bayes classifier
nb_clf.fit(X_train_tfidf, y_train)


def check_text(text):
    if len(text) == 0:
        return False

    if not re.search(r'[a-zA-Z]', text):
        return False

    return True


def predict_sentiment(text):
    if not check_text(text):
        return "Invalid"
    # Preprocess the input text
    preprocessed_text = stem_text(preprocess_text(text))

    # Transform the text using the trained tfidf_vectorizer
    features = tfidf_vectorizer.transform([preprocessed_text])

    # Predict using the trained classifier
    prediction = nb_clf.predict(features)[0]

    # Return the sentiment
    if prediction == 1:
        return "Negative"
    else:
        return "Positive"
