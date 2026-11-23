# IMDB Sentiment — Accuracy and Failure Modes

Repository for the final project in ECE4450J Introduction to Machine Learning.

## Setup

Developed with Python 3.13. Install the dependencies with:

    python -m pip install -r requirements.txt

## Data

The dataset ([stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)) is downloaded automatically from Hugging Face when the code runs, so an internet connection is required. No files need to be downloaded manually.

## Running the code

    python main.py

This does the following, in order:
1. Loads the official train and test splits
2. Prints the class balance, average review length and number of duplicates
3. Removes duplicates and test reviews from the training data
4. Splits 20% of the training data into a validation set (stratified, `RANDOM_STATE = 18`)
5. Trains the naive baseline and Naive Bayes, and prints their validation accuracy

Final Naive Bayes settings: `CountVectorizer(ngram_range=(1, 3), binary=True)` and `MultinomialNB(alpha=0.3)`.

## DistilBERT

(To be added by Mads)