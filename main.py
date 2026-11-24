import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def test_model_parameters():
    # Test different alpha values for MultinomialNB and BernoulliNB
    alphas = [0.01, 0.1, 0.3, 1.0, 3.0]
    results = []
    best_model = None
    best_accuracy = 0

    for alpha in alphas:
        multinomial_model = make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True), MultinomialNB(alpha=alpha))
        multinomial_model.fit(text_training_data, label_training_data)
        multinomial_predictions = multinomial_model.predict(text_validation_data)
        multinomial_accuracy = accuracy_score(label_validation_data, multinomial_predictions)
        results.append(("Multinomial", alpha, multinomial_accuracy))

        if multinomial_accuracy > best_accuracy:
            best_accuracy = multinomial_accuracy
            best_model = multinomial_model

        bernoulli_model = make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True), BernoulliNB(alpha=alpha))
        bernoulli_model.fit(text_training_data, label_training_data)
        bernoulli_predictions = bernoulli_model.predict(text_validation_data)
        bernoulli_accuracy = accuracy_score(label_validation_data, bernoulli_predictions)
        results.append(("Bernoulli", alpha, bernoulli_accuracy))

        if bernoulli_accuracy > best_accuracy:
            best_accuracy = bernoulli_accuracy
            best_model = bernoulli_model

    for model_type, alpha, accuracy in results:
        print(model_type, "alpha =", alpha, "accuracy =", round(accuracy, 4))

    print("Best model:", best_model, "with accuracy", round(best_accuracy, 4))

    return best_model

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

#To avoid running the model parameter testing every time, we can comment out the following line after the first run and use the best model found.
#best_model = test_model_parameters()

#Create the best pipeline model pipeline, on both the training and validation data
naive_bayes_model = make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True), MultinomialNB(alpha=0.3))
naive_bayes_model.fit(df['text'], df['label'])
