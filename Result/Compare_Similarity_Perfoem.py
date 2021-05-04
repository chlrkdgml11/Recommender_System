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


array_01 = [[0 for col in range(15)] for row in range(31)]
array_02 = [[0 for col in range(15)] for row in range(31)]


df = pd.read_csv("./Data/u.data", sep='\t', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]

for u in range(100, 951, 50):
    if(u == 950):
        u = 943
    print('u = ', u)
    n_users = u
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

    # Cosine
    k=5
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
    for i in range(n_users):
        for j in range(n_items):
            if(user_pred_k[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                cnt += 1

    cosine_rmse = np.sqrt(sum / cnt)

    # Euclidean
    k=5
    neigh = NearestNeighbors(n_neighbors=k, metric="euclidean")
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
    for i in range(n_users):
        for j in range(n_items):
            if(user_pred_k[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                cnt += 1

    euclidean_rmse = np.sqrt(sum / cnt)


    # Pearson
    k=5
    neigh = NearestNeighbors(n_neighbors=k, metric="correlation")
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
    for i in range(n_users):
        for j in range(n_items):
            if(user_pred_k[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                cnt += 1

    correlation_rmse = np.sqrt(sum / cnt)


    # Jaccard
    k=5
    neigh = NearestNeighbors(n_neighbors=k, metric="correlation")
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
    for i in range(n_users):
        for j in range(n_items):
            if(user_pred_k[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                cnt += 1

    jaccard_rmse = np.sqrt(sum / cnt)




    if(u == 100):
        plt.plot(u, cosine_rmse, 'rs', label='Cosine')
        plt.plot(u, euclidean_rmse, 'bo', label='Euclidean')
        plt.plot(u, correlation_rmse, 'yv', label='Correlation')
        plt.plot(u, jaccard_rmse, 'g^', label='Jaccard')
    else:
        plt.plot(u, cosine_rmse, 'rs')
        plt.plot(u, euclidean_rmse, 'bo')
        plt.plot(u, correlation_rmse, 'yv')
        plt.plot(u, jaccard_rmse, 'g^')



plt.legend()
plt.title('Compare Similarity')
plt.xlabel('Number of Users')
plt.ylabel('RMSE')
plt.grid()
plt.show()
plt.savefig('Compare_Similarity.png')
    