import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors
import pickle
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


array_01 = [[0 for col in range(15)] for row in range(31)]
array_02 = [[0 for col in range(15)] for row in range(31)]


df = pd.read_csv("./Data/ratings_small.csv", sep=',', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]


n_users = df.user_id.max()
n_items = 155

modified_df = df.loc[(df['user_id']  <= n_users) & (df['item_id']  <= n_items)]
print(modified_df.shape)
df_train, df_test = train_test_split(modified_df, test_size=0.2, random_state=42)
print(df_train.shape)
print(df_test.shape)


np_train = np.array(df_train)
np_test = np.array(df_test)


ratings_train = np.zeros((n_users, n_items))
for i in range(np_train.shape[0]):
    ratings_train[int(np_train[i][0])-1][int(np_train[i][1])-1] = np_train[i][2]


ratings_test = np.zeros((n_users, n_items))
for i in range(np_test.shape[0]):
    ratings_test[int(np_test[i][0])-1][int(np_test[i][1])-1] = np_test[i][2]


ratings_train = ratings_train.T
ratings_test = ratings_test.T
rmse = 100
for x in range(2,51):
    print('x = ', x)
    k=x
    neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
    neigh.fit(ratings_train)


    top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
    user_pred_k = np.zeros(ratings_train.shape)


    for i in range(ratings_train.shape[0]):
        bottom = np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T
        if(bottom[0] == 0):
            bottom[0] = 1
        user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / bottom


    sum = 0
    cnt = 0
    for i in range(n_items):
        for j in range(n_users):
            if(user_pred_k[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                cnt += 1

    if(rmse > np.sqrt(sum/cnt)):
        rmse = np.sqrt(sum/cnt)
        print(x)
        print(np.sqrt(sum/cnt))


