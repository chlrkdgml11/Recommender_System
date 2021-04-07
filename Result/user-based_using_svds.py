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
import matplotlib.pyplot as plt


array_01 = [[0 for col in range(2)] for row in range(51)]
array_02 = [[0 for col in range(2)] for row in range(51)]


df = pd.read_csv("./Data/ratings_small.csv", sep=',', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]


n_users = df.user_id.max()
n_items = 445

modified_df = df.loc[(df['user_id']  <= n_users) & (df['item_id']  <= n_items)]
df_train, df_test = train_test_split(modified_df, test_size=0.2, random_state=42)


np_train = np.array(df_train)
np_test = np.array(df_test)


ratings_train = np.zeros((n_users, n_items))
for i in range(np_train.shape[0]):
    ratings_train[int(np_train[i][0])-1][int(np_train[i][1])-1] = np_train[i][2]


ratings_test = np.zeros((n_users, n_items))
for i in range(np_test.shape[0]):
    ratings_test[int(np_test[i][0])-1][int(np_test[i][1])-1] = np_test[i][2]


rmse = 100
for x in range(1, 444):
    if(x % 50 == 0):
        print('x = ', x)
    U, sigma, Vt = svds(ratings_train, k = x)

    # print(U.shape)
    # print(sigma.shape)
    # print(Vt.shape)

    sigma = np.diag(sigma)

    pred_ratings = np.dot(np.dot(U, sigma), Vt)

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
        print(np.sqrt(sum/cnt))

    plt.plot(x, np.sqrt(sum/cnt), 'ro')

plt.savefig('svds.png')


plt.close()
    

