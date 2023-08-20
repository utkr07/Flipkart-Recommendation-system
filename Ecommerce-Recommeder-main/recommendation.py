# -*- coding: utf-8 -*-


from itertools import filterfalse
import pandas as pd

# Load the specific CSV file
df = pd.read_csv('Air Conditioners.csv')

# Add an extra column named 'pid' using the index as values
df['pid'] = df.index




df = df.drop(['image', 'link'], axis=1)

df

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['main_category'] = label_encoder.fit_transform(df['main_category'])
df['sub_category'] = label_encoder.fit_transform(df['sub_category'])
df['sub'] = label_encoder.fit_transform(df['sub_category'])

# Remove currency symbols, commas, and other non-numeric characters from 'discount_price' and 'actual_price'
df['discount_price'] = df['discount_price'].replace('[^0-9]', '', regex=True)
df['actual_price'] = df['actual_price'].replace('[^0-9]', '', regex=True)

# Convert to integers, handling NaN values
df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce', downcast='integer')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce', downcast='integer')

df

# Convert columns from string to float
df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')

# Convert columns from float to integer
df['ratings'] = df['ratings'].astype(float)
df['no_of_ratings'] = df['no_of_ratings'].astype(float)

# Calculate the mean of each column
ratings_mean = df['ratings'].mean()
no_of_ratings_mean = df['no_of_ratings'].mean()

# Replace all occurrences of -1 with the calculated mean
df['ratings'] = df['ratings'].replace(-1, ratings_mean)
df['no_of_ratings'] = df['no_of_ratings'].replace(-1, no_of_ratings_mean)



df

df['ratings'].fillna(0, inplace=True)
df['no_of_ratings'].fillna(0, inplace=True)
df['discount_price'].fillna(-1, inplace=True)
df['actual_price'].fillna(-1, inplace=True)

print(df.dtypes)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
name_tfidf = tfidf_vectorizer.fit_transform(df['name'])

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


# Lemmatized Tokens
product_name = df['name']
lemmatized_tokens = []

for name in product_name:
    tokens = word_tokenize(name)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens.append([lemmatizer.lemmatize(word) for word in filtered_tokens])

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the lemmatized tokens
lemmatized_tfidf = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in lemmatized_tokens])

# Combine TF-IDF features with other columns
other_cols = df[['main_category', 'sub_category', 'ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'pid', 'sub']].values
combined_features = np.hstack((lemmatized_tfidf.toarray(), other_cols))

# Calculate cosine similarity
cosine_sim_matrix = cosine_similarity(combined_features)

# Recommendation function
def recommend_similar_products(product_indices, top_n=10):
    avg_feature = combined_features[product_indices].mean(axis=0)
    similarities = cosine_similarity(avg_feature.reshape(1, -1), combined_features)
    similar_product_indices = similarities.argsort()[0][::-1][:top_n]
    return similar_product_indices

input_product_index = 5
recommended_indices = recommend_similar_products([input_product_index], top_n=10)

# Print input product name and recommended product names
print("Input Product:")
print(df['name'][input_product_index])

print("\nRecommended Products:")
for index in recommended_indices:
    print(df['name'][index])


import pickle
pickle.dump(df.to_dict(),open('questions.pkl','wb'))
pickle.dump(combined_features,open('similarity.pkl','wb'))








