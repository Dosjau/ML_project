import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

#Import the dataset from Hugging Face
splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])

#Data analysis
print("Class balance (training set):")
print(df['label'].value_counts())

review_length = df['text'].str.split().str.len()
print("Average review length (words):", review_length.mean())

print("Duplicates in training data:", df['text'].duplicated().sum())
print("Test reviews also in training data:", df_test['text'].isin(df['text']).sum())

#Remove duplicates from the training data
df = df.drop_duplicates(subset='text')
df = df[~df['text'].isin(df_test['text'])]

# Split the data
RANDOM_STATE = 18
text_training_data, text_validation_data, label_training_data, label_validation_data = train_test_split(
    df['text'],
    df['label'],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df['label']
)

print("Train / validation / test:", len(text_training_data), len(text_validation_data), len(df_test))

#Create naive baseline
positive_count = np.sum(label_training_data == 1)
negative_count = np.sum(label_training_data == 0)

if positive_count > negative_count:
    naive_baseline_prediction = np.ones(len(label_validation_data))
else:
    naive_baseline_prediction = np.zeros(len(label_validation_data))

#Create model pipeline
naive_bayes_multinomial_model = make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True), MultinomialNB(alpha=0.3))
naive_bayes_multinomial_model.fit(text_training_data, label_training_data)

validation_data_prediction_multinomial = naive_bayes_multinomial_model.predict(text_validation_data)

#Calculate accuracy
naive_baseline_accuracy = accuracy_score(label_validation_data, naive_baseline_prediction)
naive_bayes_accuracy_multinomial = accuracy_score(label_validation_data, validation_data_prediction_multinomial)

print("Naive Baseline Accuracy:", naive_baseline_accuracy)
print("Naive Bayes (Multinomial) Accuracy:", naive_bayes_accuracy_multinomial)
