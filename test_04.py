import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('./Data/u.data', sep='\t', header=None)

# print(df.head())

df.columns = ["user_id", "item_id", "rating", "timestamp"]

# print(df.head())

n_users = df.user_id.max()
n_items = df.item_id.max()

# print(n_users, n_items)

ratings = np.zeros((n_users, n_items))

for row in df.itertuples():
    ratings[row[1]-1,row[2]-1] = row[3]

ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)

item_distances = cosine_similarity(ratings_train.T)

print('item_distances\n', item_distances)
print(item_distances.shape)

item_pred = item_distances.dot(ratings_train.T) / np.array([np.abs(item_distances).sum(axis=1)]).T

print('item_pred\n', item_pred)
print(item_pred.shape)

k=5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
neigh.fit(ratings_train)

top_k_distances, top_k_items = neigh.kneighbors(ratings_train, return_distance=True)

item_pred_k = np.zeros(ratings_train.T.shape)

for i in range(ratings_train.shape[1]):
    if(i%50==0):
        print("cnt = ", i)
    item_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_items][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print('item_pred_k\n', item_pred_k)
print('item_pred_k.shape\n', item_pred_k.shape)
