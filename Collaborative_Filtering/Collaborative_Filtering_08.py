import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('../Data/u.data', sep='\t', header=None)

# print(df.head())

df.columns = ["user_id", "item_id", "rating", "timestamp"]

# print(df.head())

n_users = df.user_id.max()
n_items = df.item_id.max()

# print(n_users, n_items)

ratings = np.zeros((n_users, n_items))

for row in df.itertuples():
    ratings[row[1]-1,row[2]-1] = row[3]

# ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)
df_train = pd.read_csv('../Data/u1.base', sep='\t', header=None)
df_train.columns = ["user_id", "item_id", "rating", "timestamp"]

n_users = df_train.user_id.max()
n_items = df_train.item_id.max()

ratings_train = np.zeros((n_users, n_items))

for row in df_train.itertuples():
    ratings_train[row[1]-1,row[2]-1] = row[3]


df_test = pd.read_csv('../Data/u1.test', sep='\t', header=None)
df_test.columns = ["user_id", "item_id", "rating", "timestamp"]

n_users = df_test.user_id.max()
n_items = df_test.item_id.max()

ratings_test = np.zeros((n_users, n_items))

for row in df_test.itertuples():
    ratings_test[row[1]-1,row[2]-1] = row[3]

user_distances = cosine_similarity(ratings_train)


# print('ratings_train\n', ratings_train)
# print('ratings_train.shape\n', ratings_train.shape)
# print('ratings_test\n', ratings_test)
# print('ratings_test.shape\n', ratings_test.shape)
# print('user_distances\n', user_distances)
# print('user_distances.shape\n', user_distances.shape)


user_pred = user_distances.dot(ratings_train) / np.array([np.abs(user_distances).sum(axis=1)]).T

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


k=ratings_train.shape[1]
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
neigh.fit(ratings_train.T)

item_distances, _= neigh.kneighbors(ratings_train.T, return_distance=True)

print('item_distances\n', item_distances)

print(item_distances.shape)

item_pred = ratings_train.dot(item_distances) / np.array([np.abs(item_distances).sum(axis=1)])


print('item_pred\n', item_pred)
print(item_pred.shape)

print('RMSE\n', np.sqrt(get_mse(item_pred, ratings_test)))
