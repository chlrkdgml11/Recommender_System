import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('../Data/ratings_small.csv')

print(data.head())

data = data.pivot_table('rating', index = 'userId', columns = 'movieId')

ratings = pd.read_csv('../Data/ratings_small.csv')
movies = pd.read_csv('../Data/tmdb_5000_movies.csv')

movies.rename(columns = {'id': 'movieId'}, inplace = True)

ratings_movies = pd.merge(ratings, movies, on = 'movieId')

print(ratings_movies.head(1))

data = ratings_movies.pivot_table('rating', index = 'userId', columns = 'title').fillna(0)

data = data.transpose()

movie_sim = cosine_similarity(data, data)
print(movie_sim.shape)

movie_sim_df = pd.DataFrame(data = movie_sim, index = data.index, columns = data.index)

print(movie_sim_df["Harry Potter and the Half-Blood Prince"].sort_values(ascending=False)[:10])
