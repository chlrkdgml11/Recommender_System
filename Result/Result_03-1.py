import pandas as pd
import numpy as np

N_user = 943
N_movie = 1682

df_train = pd.read_csv('../Data/u1.base', sep='\t', header=None)
df_train.columns = ["user_id", "item_id", "rating", "timestamp"]
del df_train["timestamp"]
n_users = df_train.user_id.max()
n_movies = 1682
ratings_train = np.zeros((n_users, n_movies))
df_train = np.array(df_train)
for i in range(df_train.shape[0]):
    ratings_train[df_train[i][0]-1][df_train[i][1]-1] = df_train[i][2]

# neighbors = { user_id : [sorted (user_id, similarity)] }
neighbors = {}
# means = { user_id : mean of user's ratings}
means = [[0 for col in range(2)] for row in range(943)]
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
    
    # 두 사람이 본 영화만 리스트로 저장
    for movie_id in ratings_train[user1-1]:
        if movie_id in ratings_train[user2-1]:
            rating1.append(ratings_train[user1-1][movie_id])
            rating2.append(ratings_train[user2-1][movie_id])
    
    if len(rating1) == 0:
        return 0.0
    
    for x in range(len(rating1)):
        rating1[x] = rating1[x] - means[user1-1]
        rating2[x] = rating2[x] - means[user2-1] #각 평점에서 평점의 평균을 빼준다.

    vec = ((np.linalg.norm(rating1))*(np.linalg.norm(rating2)))

    if vec != 0.0:
        sim = np.dot(rating1,rating2)/(vec)#코사인 유사도 계산
    
        return round(sim,4)#소수점 아래 4자리

    else:
        return 0.0
        
print(sim_user(1,2))
print(sim_user(1,3))
print(sim_user(1,4))
'''
def calculate_similarity():
    print('N_user+1 = ', N_user+1)
    for i in range(1, N_user+1):
        if(i % 50 == 0):
            print('i = ', i)
        nei = []
        for j in range(1, N_user+1):
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
    

def prediction():
    test = pd.read_csv("../Data/u1.test", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
    del test['timestamp']
    test = np.array(test)
    
    sqaure = 0
    for user_id, movie_id, rating in test:
        sqaure += (predict_rating(user_id, movie_id) - rating) ** 2

    return np.sqrt(sqaure / len(test))


rmse = prediction()
print(rmse)
'''
