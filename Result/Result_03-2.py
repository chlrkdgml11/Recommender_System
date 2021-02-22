import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

array_01 = [[0 for col in range(2)] for row in range(25)]
array_02 = [[0 for col in range(2)] for row in range(25)]


for x in range(2,21):
    df = pd.read_csv("./Data/u.data", sep='\t', header=None)
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
        ratings_train[np_train[i][0]-1][np_train[i][1]-1] = np_train[i][2]


    ratings_test = np.zeros((n_users, n_items))
    for i in range(np_test.shape[0]):
        if(np_test[i][1] > 600):
            continue
        ratings_test[np_test[i][0]-1][np_test[i][1]-1] = np_test[i][2]


    means = [[0 for col in range(2)] for row in range(n_users)]
    for i in range(len(ratings_train)):
        rat = 0
        count = 0
        for j in range(len(ratings_train[0])):
            if(ratings_train[i][j] == 0):
                continue
        count += 1
        rat += ratings_train[i][j]
        means[i] = rat / count    


    def sim_user(user1,user2):
        rating1 = []
        rating2 = []
        
        for i in range(n_items):
            if(ratings_test[user1][i] != 0 and ratings_test[user2][i] !=0):
                rating1.append(ratings_test[user1][i])
                rating2.append(ratings_test[user2][i]) 
        
        if len(rating1) == 0:
            return 0.0

        vec = ((np.linalg.norm(rating1))*(np.linalg.norm(rating2)))

        if vec != 0.0:
            sim = np.dot(rating1,rating2)/(vec)#코사인 유사도 계산
            return round(sim,4)#소수점 아래 4자리


    neighbors = {}
    def calculate_similarity():
        print('n_users = ', n_users)
        for i in range(n_users):
            if(i % 50 == 0):
                print('i = ', i)
            nei = []
            for j in range(n_users):
                if i != j:
                    nei.append((j, sim_user(i, j)))
                    
            # 내림차순으로 정렬            
            nei.sort(key=lambda x: x[1], reverse=True)
            neighbors[i] = nei

    calculate_similarity()   # 모든 유저간 유사도 계산해서 저장


    def predict_rating(user_id, movie_id):
        rating = 0
        K = 0
        j = 0
        for i in range(N_user - 1):
            if j > 5:
                break
            # 해당 영화 평점을 실제로 매긴 neighbor 유저만 취급
            if movie_id in ratings[neighbors[user_id][i][0]]:
                j += 1
                nei_id = neighbors[user_id][i][0]
                nei_sim = neighbors[user_id][i][1]

                rating += ratings[nei_id][movie_id] * nei_sim
                K += nei_sim
        
        if K != 0:
            rating = (rating / K) 
        else:
            rating = 2.3
            
        return rating








    user_distances = np.zeros((n_users,n_users))
    for i in range(n_users):
        if(i % 10 == 0):
            print('i = ', i)
        for j in range(n_users):
            user_distances[i][j] = sim_user(i, j)
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
    print(array_01)

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

print('Result_02-2')
print(array_01)
print(array_02)