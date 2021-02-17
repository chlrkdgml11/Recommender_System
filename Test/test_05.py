import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

df_train = pd.read_csv('../Data/u1.base', sep='\t', header=None)
df_train.columns = ["user_id", "item_id", "rating", "timestamp"]
del df_train["timestamp"]
n_users = df_train.user_id.max()
n_items = 1682
ratings_train = np.zeros((n_users, n_items))
df_train = np.array(df_train)
for i in range(df_train.shape[0]):
    ratings_train[df_train[i][0]-1][df_train[i][1]-1] = df_train[i][2]

df_test = pd.read_csv('../Data/u1.test', sep='\t', header=None)
df_test.columns = ["user_id", "item_id", "rating", "timestamp"]
del df_test["timestamp"]
n_users = df_test.user_id.max()
n_items = 1682
ratings_test = np.zeros((n_users, n_items))
df_test = np.array(df_test)
for i in range(df_test.shape[0]):
    ratings_test[df_test[i][0]-1][df_test[i][1]-1] = df_test[i][2]


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


print('RMSE\n', np.sqrt(get_mse(user_pred, ratings_test)))


k=5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
neigh.fit(ratings_train)


top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
user_pred_k = np.zeros(ratings_train.shape)


for i in range(ratings_train.shape[0]):
    if(i%100==0):
        print("cnt = ", i)
    user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T


print('RMSE\n', np.sqrt(get_mse(user_pred_k, ratings_test)))
