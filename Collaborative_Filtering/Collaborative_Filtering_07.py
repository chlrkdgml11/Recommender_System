import pandas as pd
import numpy as np

N_user = 943
N_movie = 1682

data = pd.read_csv("../Data/u1.base", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
del data['timestamp']
# data = np.array(data)

ratings = np.zeros((N_user, N_movie))
for row in data.itertuples():
    ratings[row[1]-1,row[2]-1] = row[3]


# neighbors = { movie_id : [sorted (movie_id, similarity)] }
neighbors = {}
# means = { movie_id : mean of movie_id's ratings}
means = np.zeros((N_movie+1, 2))
print(means)
print(means.shape)
for i in range(len(ratings[0])):
    rat = 0
    count = 0
    for j in range(len(ratings)):
        if(ratings[j][i]):
            count += 1
            rat += ratings[j][i]
    if(count):
        means[i+1][0] = i+1
        means[i+1][1] = rat / count
    if(count == 0):
        print('i = ',i) 

for i in range(len(means)):
    print(means[i])

def sim_item(item1,item2):
       
    rating1 = []
    rating2 = []


    # 두 영화를 모두본 사용자만 리스트로 저장
    for user_id in np_ratings[:, item1]:
        if user_id in np_ratings[:, item2]:
            rating1.append(np_ratings[user_id][item1])
            rating2.append(np_ratings[user_id][item2])
    
    # print('rating1\n', rating1)
    # print('rating2\n', rating2)
    if len(rating1) == 0:
        return 0.0
    
    for x in range(len(rating1)):
        rating1[x] = rating1[x] - means[item1]
        rating2[x] = rating2[x] - means[item2] #각 평점에서 평점의 평균을 빼준다.

    vec = ((np.linalg.norm(rating1))*(np.linalg.norm(rating2)))

    if vec != 0.0:
        sim = np.dot(rating1,rating2)/(vec)#코사인 유사도 계산
    
        return round(sim,4)#소수점 아래 4자리

    else:
        return 0.0

for i in range(1, N_movie+1):
    if(i % 50 == 0):
        print('i = ', i)
    nei = []
    for j in range(1, N_movie+1):
        if i != j:
            nei.append((j, sim_item(i, j)))

def calculate_similarity():
    print('N_movie+1 = ', N_movie+1)
    for i in range(1, N_movie+1):
        if(i % 50 == 0):
            print('i = ', i)
        nei = []
        for j in range(1, N_movie+1):
            if i != j:
                nei.append((j, sim_item(i, j)))
                   
        # 내림차순으로 정렬            
        nei.sort(key=lambda x: x[1], reverse=True)
        neighbors[i] = nei

calculate_similarity()   # 모든 영화간 유사도 계산해서 저장


# def predict_rating(user_id, movie_id):
#     rating = 0
#     K = 0
#     j = 0
#     for i in range(N_user - 1):
#         # valid neighbor 40개까지
#         if j > 5:
#             break
#         # 해당 영화 평점을 실제로 매긴 neighbor 유저만 취급
#         if movie_id in ratings[neighbors[user_id][i][0]]:
#             j += 1
#             nei_id = neighbors[user_id][i][0]
#             nei_sim = neighbors[user_id][i][1]
#                                                  ### mean으로 보정
#             rating += (ratings[nei_id][movie_id] - means[nei_id]) * nei_sim
#             K += nei_sim
    
#     if K != 0:
#         rating = means[user_id] + (rating / K)  ###
#     else:
#         rating = 2.3
        
#     return rating


# def prediction():
#     test = pd.read_csv("./Data/u1.test", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
#     del test['timestamp']
#     test = np.array(test)
    
#     sqaure = 0
#     #  RMSE (Root Mean Square Error) 
#     for user_id, movie_id, rating in test:
#         # sum of (아이템의 예측 레이팅 - 아이템의 원래 레이팅) ** 2
#         sqaure += (predict_rating(user_id, movie_id) - rating) ** 2
#     return np.sqrt(sqaure / len(test))

# rmse = prediction()
# print(rmse)
