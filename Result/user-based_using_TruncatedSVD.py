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
from sklearn.decomposition import TruncatedSVD


array_01 = [[0 for col in range(2)] for row in range(51)]
array_02 = [[0 for col in range(2)] for row in range(51)]


df = pd.read_csv("./Data/ratings_small.csv", sep=',', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]


n_users = df.user_id.max()
n_items = 445

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

user_ratings_mean = np.mean(ratings_train, axis = 1)

rmse = 100
for x in range(2,51):
    print('x = ', x)
    for y in range(2,51):
        svd = TruncatedSVD(n_components = x, n_iter = y, random_state = 42)
        
        US = svd.fit_transform(ratings_train)
        V = svd.components_
        S = svd.singular_values_

        pred_ratings = np.dot(US, V)

        sum = 0
        cnt = 0
        for i in range(n_users):
            for j in range(n_items):
                if(pred_ratings[i][j] != 0 and ratings_test[i][j] != 0):
                    sum += (pred_ratings[i][j] - ratings_test[i][j]) ** 2
                    cnt += 1

        if(rmse > np.sqrt(sum/cnt)):
            rmse = np.sqrt(sum/cnt)
            print(x)
            print(y)
            print(np.sqrt(sum/cnt))
