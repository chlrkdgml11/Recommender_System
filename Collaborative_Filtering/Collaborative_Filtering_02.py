import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors


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

# ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)

user_distances = cosine_similarity(ratings_train)

user_pred = user_distances.dot(ratings_train) / np.array([np.abs(user_distances).sum(axis=1)]).T


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    print('actual.nonzero()\n', actual.nonzero())
    print(pred)
    print(actual)
    print(len(pred))
    print(len(actual))
    return mean_squared_error(pred, actual)

print('user_pred\n', user_pred)
print('ratings_test\n', ratings_test)

print('RMSE\n', np.sqrt(get_mse(user_pred, ratings_test)))

k=5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
neigh.fit(ratings_train)


top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)

user_pred_k = np.zeros(ratings_train.shape)


print(top_k_distances.T.shape)
print(ratings_train.shape)
print(top_k_distances.shape)


for i in range(ratings_train.shape[0]):
    if(i%100==0):
        print("cnt = ", i)
    user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print(user_pred_k.shape)
print(ratings_test.shape)

print('RMSE\n', np.sqrt(get_mse(user_pred_k, ratings_test)))


