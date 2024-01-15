# Amazon Sentiment Analysis with NLTK and Naive Bayes

## Overview

This project is a sentiment analysis tool designed to classify Amazon customer reviews as either positive or negative. The implementation utilizes the Natural Language Toolkit (NLTK) for text processing, a Naive Bayes classifier for sentiment analysis, and Tkinter for the graphical user interface (GUI) frontend.


## Features

### 1. Libraries Used:

The project utilizes the following libraries:
- `nltk`: Natural Language Toolkit for natural language processing tasks.
- `pandas`: Data manipulation and analysis library.
- `re`: Regular expression library for text preprocessing.
- `sklearn`: Machine learning library for TF-IDF vectorization and Naive Bayes classification.
- `warnings`: Used for managing warnings during code execution.

### 2. Data Preprocessing:

The dataset undergoes preprocessing to ensure effective sentiment analysis:
- **Lowercasing:** Reviews are converted to lowercase for consistency.
- **Text Cleaning:** Removal of URLs, square brackets, non-alphabetic characters, and stop words from reviews.

### 3. Tokenization and Stemming:

The reviews are tokenized into words, and stemming (reducing words to their base form) is applied to each word.

### 4. Word Frequency Analysis:

A word frequency analysis is conducted to understand the most common words in the dataset. The top words are ranked by frequency.

### 5. TF-IDF Vectorization:

The TF-IDF vectorizer is instantiated with a maximum of 5000 features and a n-gram range of (1, 2). This is used to transform the reviews into a numerical format suitable for machine learning.

### 6. Naive Bayes Classification:

A Multinomial Naive Bayes classifier is trained using the TF-IDF-transformed data and review polarities.

### 7. Tkinter Frontend:

The project includes a Tkinter-based GUI for a user-friendly experience. Users can input reviews and obtain sentiment predictions.


## Contributing

We welcome contributions from the community! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your fork.
5. Submit a pull request.

