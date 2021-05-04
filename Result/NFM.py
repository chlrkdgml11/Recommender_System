import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import svd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import correlation
from sklearn.decomposition import NMF




df = pd.read_csv("./Data/u.data", sep='\t', header=None)
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


result = 10
for x in range(21,41):
    nmf = NMF(n_components=x)

    nmf.fit_transform(ratings_train)

    W = nmf.fit_transform(ratings_train)

    H = nmf.components_

    pred_ratings = np.dot(W, H)



    # for i in range(n_users):
    #     for j in range(n_items):
    #         if(pred_ratings[i][j] >= 5):
    #             pred_ratings[i][j] = 5


    sum = 0
    cnt = 0

    for i in range(n_users):
        for j in range(n_items):
            if(pred_ratings[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (pred_ratings[i][j] - ratings_test[i][j]) ** 2
                cnt += 1
    

    if(cnt != 0):
        rmse = np.sqrt(sum / cnt)
        if(result > rmse):
            result = rmse
            print('x = ', x)
            print(result)



