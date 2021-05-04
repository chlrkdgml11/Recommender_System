import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("./Data/u.data", sep='\t', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]


n_users = df.user_id.max()
n_items = 445

modified_df = df.loc[(df['user_id']  <= n_users) & (df['item_id']  <= n_items)]
df_train, df_test = train_test_split(modified_df, test_size=0.2, random_state=42)




np_test = np.array(df_test)
ratings_test = np.zeros((n_users, n_items))
for i in range(np_test.shape[0]):
    ratings_test[int(np_test[i][0])-1][int(np_test[i][1])-1] = np_test[i][2]


'''
arr = np.zeros((944,2))

for i in range(1,944):
    if((len(df_train.loc[df_train['user_id'] == i]) >= 20) & (len(df_train.loc[df_train['user_id'] == i]) <= 40)):
        arr[i][0] = i
        arr[i][1] = len(df_train.loc[df_train['user_id'] == i])

np.savetxt('./Find_Distinct_Point/over_20_data_users.txt', arr, fmt = '%2d', delimiter = ',', header='test')
'''

a = np.loadtxt("./Find_Distinct_Point/over_20_data_users.txt", delimiter = ',')

for i in range(len(a)):
    del_user = int(a[i][0])


    df_only_delected_user = df_train.loc[df_train['user_id'] == del_user]       # 유저 데이터 다 지운 index values 저장
    df_del_train = df_train.loc[df_train['user_id'] != del_user]      # 유저의 데이터 다 지움


    np_train = np.array(df_del_train)
    ratings_train = np.zeros((n_users, n_items))
    for i in range(np_train.shape[0]):
        ratings_train[int(np_train[i][0])-1][int(np_train[i][1])-1] = np_train[i][2]

    rmse_arr = np.zeros((df_only_delected_user.shape[0],2))
    print(df_only_delected_user.shape[0])
    for x in range(df_only_delected_user.shape[0]):
        print('x = ', x)
        print(int(del_user), int(df_only_delected_user.item_id.values[x]), df_only_delected_user.rating.values[x])
        ratings_train[int(del_user)-1][int(df_only_delected_user.item_id.values[x])-1] = df_only_delected_user.rating.values[x]


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
        for i in range(del_user-1, del_user):
            for j in range(n_items):
                if((user_pred_k[i][j] != 0) & (ratings_test[i][j] != 0)):
                    sum += (user_pred_k[i][j] - ratings_test[i][j]) ** 2
                    cnt += 1

        if(cnt == 0):
            rmse = 5
        else:
            rmse = sum / cnt

        rmse_arr[x][0] = x
        rmse_arr[x][1] = np.sqrt(rmse)

    print(rmse_arr)
    np.savetxt('./Find_Variance_Text/rmse_%d.txt' %del_user, rmse_arr, fmt = '%f', delimiter = '\t', header='test')


