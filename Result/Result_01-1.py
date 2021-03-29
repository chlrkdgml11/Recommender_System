import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

array_01 = [[0 for col in range(15)] for row in range(15)]
array_02 = [[0 for col in range(15)] for row in range(15)]



df = pd.read_csv("./Data/280,000_users_ratings.csv", sep=',', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]

for x in range(1, 2):
    for y in range(3,8,2):
        print('x = ', x)
        print('y = ', y)
        n_users = 5000 * x
        n_items = 444

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
        
        user_distances = cosine_similarity(ratings_train)
        bottom = np.array([np.abs(user_distances).sum(axis=1)]).T
        for i in range(len(bottom)):
            if(bottom[i][0] == 0):
                bottom [i][0] = 1
        user_pred = user_distances.dot(ratings_train) / bottom
        # user_pred = user_distances.dot(ratings_train) / np.array([np.abs(user_distances).sum(axis=1)]).T

        sum = 0
        cnt = 0
        for i in range(n_users):
            for j in range(n_items):
                if(user_pred[i][j] != 0 and ratings_test[i][j] != 0):
                    sum += (user_pred[i][j] - ratings_test[i][j]) ** 2
                    cnt += 1
        array_01[x][y] = np.sqrt(sum/cnt)

        k=y
        neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
        neigh.fit(ratings_train)


        top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
        user_pred_k = np.zeros(ratings_train.shape)


        for i in range(ratings_train.shape[0]):
            if(i%100==0):
                print("cnt = ", i)
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
        array_02[x][y] = np.sqrt(sum/cnt)
        print(array_01)
        print(array_02)

print('Result_01-1')
print(array_01)
print(array_02)
