import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

array = [[0 for col in range(2)] for row in range(35)]

for x in range(2,31):
    print('x = ', x)
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
        if(np_train[i][1] > n_items):
            continue
        ratings_train[np_train[i][0]-1][np_train[i][1]-1] = np_train[i][2]


    ratings_test = np.zeros((n_users, n_items))
    for i in range(np_test.shape[0]):
        if(np_test[i][1] > n_items):
            continue
        ratings_test[np_test[i][0]-1][np_test[i][1]-1] = np_test[i][2]

    # neighbors = { user_id : [sorted (user_id, similarity)] }
    neighbors = {}
    # means = { user_id : mean of user's ratings}
    means = [[0 for col in range(2)] for row in range(n_users)]
    for i in range(len(ratings_train)):
        rat = 0
        count = 0
        for j in range(len(ratings_train[0])):
            count += 1
            rat += ratings_train[i][j]
        means[i] = rat / count

  

    def sim_user(user1,user2):
        
        rating1 = []
        rating2 = [] #두 유저의 평점을 저장할 리스트
        
        #두 사람이 모두 본 영화만 리스트로 저장
        for j in range(n_items):
            if(ratings_train[user1][j] != 0 and ratings_train[user2][j] != 0):
                rating1.append(ratings_train[user1][j])
                rating2.append(ratings_train[user2][j])


        
        if len(rating1) == 0:
            return 0.0
        
        for x in range(len(rating1)):
            rating1[x] = rating1[x] - means[user1]
            rating2[x] = rating2[x] - means[user2] #각 평점에서 평점의 평균을 빼준다.

        vec = ((np.linalg.norm(rating1))*(np.linalg.norm(rating2))) #두 평점의 크기를 미리 곱한다.

        if vec != 0.0: #분모가 0이 아니라면
            sim = np.dot(rating1,rating2)/(vec)#코사인 유사도 계산
        
            return round(sim,4)#소수점 아래 4자리까지 계산후 리턴

        else:
            return 0.0#분모가 0이면 계산할 수 없다. 0.0리턴

    def calculate_similarity():
        
        # 1 ~ 943(N_user)
        for i in range(n_users):
            if(i % 50 == 0):
                print('i = ', i)
            nei = []
            for j in range(n_users):
                nei.append((j, sim_user(i, j)))
                    
            # 네이버 유저의 similarity 기준 내림차순으로 정렬            
            nei.sort(key=lambda x: x[1], reverse=True)
            neighbors[i] = nei
            # neighbors = { user_id : [sorted (user_id, similarity)] }

    calculate_similarity()   # 모든 유저간 유사도 계산해서 저장

    def predict_rating(user_id, movie_id):
        rating = 0
        K = 0
        j = 0
        for i in range(n_users):
            # valid neighbor x개까지
            if j > x:
                break
            # 해당 영화 평점을 실제로 매긴 neighbor 유저만 취급
            if movie_id in ratings_train[neighbors[user_id][i][0]]:
                j += 1
                nei_id = neighbors[user_id][i][0]
                nei_sim = neighbors[user_id][i][1]
                
                rating += (ratings_train[nei_id][movie_id] - means[nei_id]) * nei_sim
                K += nei_sim
        
        if K != 0:
            rating = (rating / K) 
        else:
            rating = 0
            
        return rating

    def prediction():
        sum = 0
        cnt = 0
        for i in range(n_users):
            if(i % 100 == 0):
                print('i = ', i)
            for j in range(n_items):
                if(predict_rating(i, j) != 0 and ratings_test[i][j] != 0):
                    sum += (predict_rating(i, j) - ratings_test[i][j]) ** 2
                    cnt += 1
        return np.sqrt(sum / cnt)




    array[x][0] = x
    array[x][1] = prediction()

print('Result_06-2')
print(array)
