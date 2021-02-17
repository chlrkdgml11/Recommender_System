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

ratings_test = ratings_test.T

user_distances = cosine_similarity(ratings_train)


# print('ratings_train\n', ratings_train)
# print('ratings_train.shape\n', ratings_train.shape)
# print('ratings_test\n', ratings_test)
# print('ratings_test.shape\n', ratings_test.shape)
# print('user_distances\n', user_distances)
# print('user_distances.shape\n', user_distances.shape)


# user_pred = user_distances.dot(ratings_train) / np.array([np.abs(user_distances).sum(axis=1)]).T

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


k=5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
neigh.fit(ratings_train)

top_k_distances, top_k_items = neigh.kneighbors(ratings_train, return_distance=True)

item_pred_k = np.zeros(ratings_train.shape)

print(top_k_distances.T.shape)
print(ratings_train.shape)
print(top_k_distances.shape)
print(item_pred_k.shape)


for i in range(ratings_train.shape[0]):
    if(i%100==0):
        print("cnt = ", i)
    item_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_items][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print(item_pred_k.shape)
print('RMSE\n', np.sqrt(get_mse(item_pred_k, ratings_test)))
