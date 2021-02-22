import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

array_01 = [[0 for col in range(2)] for row in range(25)]
array_02 = [[0 for col in range(2)] for row in range(25)]


for x in range(2,6):
    print('x = ', x)
    df = pd.read_csv("./Data/ratings_small.csv", sep=',', header=None)
    df.columns = ["user_id", "item_id", "rating", 'timestamp']
    del df["timestamp"]


    n_users = df.user_id.max()
    n_items = 600


    df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)


    np_train = np.array(df_train)
    np_test = np.array(df_test)


    ratings_train = np.zeros((n_users, n_items))
    for i in range(np_train.shape[0]):
        if(np_train[i][1] > 600):
            continue
        ratings_train[int(np_train[i][0])-1][int(np_train[i][1])-1] = np_train[i][2]


    ratings_test = np.zeros((n_users, n_items))
    for i in range(np_test.shape[0]):
        if(np_test[i][1] > 600):
            continue
        ratings_test[int(np_test[i][0])-1][int(np_test[i][1])-1] = np_test[i][2]

    user_distances = cosine_similarity(ratings_train)

    # for i in range(np.array([np.abs(user_distances).sum(axis=1)]).T.shape[0]):
    #     if(np.array([np.abs(user_distances).sum(axis=1)]).T[i][0] == 0):
    #         print(i)
    user_pred = user_distances.dot(ratings_train) / np.array([np.abs(user_distances).sum(axis=1)]).T
    # print(np.sqrt(mean_squared_error(user_pred, ratings_test)))
    sum = 0
    cnt = 0
    for i in range(n_users):
        for j in range(n_items):
            if(user_pred[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred[i][j] - ratings_test[i][j]) ** 2
                cnt += 1
    print(np.sqrt(sum/cnt))
    array_01[x][0] = x
    array_01[x][1] = np.sqrt(sum/cnt)

    k=x
    neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
    neigh.fit(ratings_train)


    top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
    user_pred_k = np.zeros(ratings_train.shape)


    for i in range(ratings_train.shape[0]):
        if(i%100==0):
            print("cnt = ", i)
        user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T


    # print(np.sqrt(mean_squared_error(user_pred_k, ratings_test)))

    sum = 0
    cnt = 0
    for i in range(n_users):
        for j in range(n_items):
            if(user_pred_k[i][j] != 0 and ratings_test[i][j] != 0):
                sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                cnt += 1
    print(np.sqrt(sum/cnt))
    array_02[x][0] = x
    array_02[x][1] = np.sqrt(sum/cnt)

print('Result_01-2')
print(array_01)
print(array_02)