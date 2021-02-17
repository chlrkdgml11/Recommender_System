import pandas as pd
import numpy as np

data = pd.read_csv('./Data/u1.base', sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
del data['timestamp']

N_user = data["userId"].max()
N_movie = data["movieId"].max()

print('N_user', N_user)
print('N_movie', N_movie)

data = np.array(data)

# ratings = { user_id : { movie_id: rating } }
ratings = {i:{} for i in range(1,N_user+1)}
for a,b,c in data:
    ratings[a][b] = c

neighbors = {}
means = {}

for i in ratings:
    rat = 0
    count = 0
    for j in ratings[i]:
        count += 1
        rat += ratings[i][j]
    means[i] = rat / count    


def sim_user(user1,user2):
       
    rating1 = []
    rating2 = [] #두 유저의 평점을 저장할 리스트
    
    #두 사람이 모두 본 영화만 리스트로 저장
    for movie_id in ratings[user1]:
        if movie_id in ratings[user2]:
            rating1.append(ratings[user1][movie_id])
            rating2.append(ratings[user2][movie_id])
      
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

print(sim_user(1,2))
print(sim_user(1,3))
print(sim_user(1,4))


def calculate_similarity():
    
    # 1 ~ 943(N_user)
    print(N_user)
    for i in range(1, N_user+1):
        if(i % 50 == 0):
            print('i = ', i)
        nei = []
        for j in range(1, N_user+1):
            if i != j:
                nei.append((j, sim_user(i, j)))
                   
        # 네이버 유저의 similarity 기준 내림차순으로 정렬            
        nei.sort(key=lambda x: x[1], reverse=True)
        neighbors[i] = nei
        # neighbors = { user_id : [sorted (user_id, similarity)] }


calculate_similarity()   # 모든 유저간 유사도 계산해서 저장
print(len(neighbors[2])) # 유저2의 다른 942명 유저와의 유사도
print(neighbors[2])


def predict_rating(user_id, movie_id):
    rating = 0
    K = 0
    j = 0
    for i in range(N_user - 1):
        # valid neighbor 40개까지
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


print(predict_rating(1,1))
print(predict_rating(1,2))

def prediction():
    test = pd.read_csv("./Data/u1.test", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
    del test['timestamp']
    test = np.array(test)
    
    sqaure = 0
    #  RMSE (Root Mean Square Error) 
    for user_id, movie_id, rating in test:
        # sum of (아이템의 예측 레이팅 - 아이템의 원래 레이팅) ** 2
        sqaure += (predict_rating(user_id, movie_id) - rating) ** 2
    return np.sqrt(sqaure / len(test))

print(predict_rating(1,1))
print(predict_rating(1,2))

def prediction_01():
    test = pd.read_csv("./Data/u1.base", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
    del test['timestamp']
    test = np.array(test)
    
    sqaure = 0
    #  RMSE (Root Mean Square Error) 
    for user_id, movie_id, rating in test:
        # sum of (아이템의 예측 레이팅 - 아이템의 원래 레이팅) ** 2
        sqaure += (predict_rating(user_id, movie_id) - rating) ** 2
    return np.sqrt(sqaure / len(test))

rmse_01 = prediction()
rmse_02 = prediction_01()

print(rmse_01)
print(rmse_02)


