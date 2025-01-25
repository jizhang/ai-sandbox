import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from ch02_sentiment import get_embedding, cosine_similarity

datafile_path = os.path.join(os.getenv('SAMPLE_DATA_DIR'), 'fine_food_reviews_with_embeddings_1k.csv')
df = pd.read_csv(datafile_path)
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].replace({
    1: 'negative',
    2: 'negative',
    4: 'positive',
    5: 'positive',
})

labels = [
    'An Amazon review with a negative sentiment.',
    'An Amazon review with a positive sentiment.',
]
label_embeddings = [get_embedding(label) for label in labels]


def label_score(review_embedding, label_embeddings):
    positive_similarity = cosine_similarity(review_embedding, label_embeddings[1])
    negative_similarity = cosine_similarity(review_embedding, label_embeddings[0])
    return positive_similarity - negative_similarity


probas = df['embedding'].apply(lambda x: label_score(x, label_embeddings))
preds = probas.apply(lambda x: 'positive' if x > 0 else 'negative')

report = classification_report(df['sentiment'], preds)
print(report)