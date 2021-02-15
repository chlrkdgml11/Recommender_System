import numpy as np
from surprise import Dataset

data = Dataset.load_builtin('ml-100k', prompt=False)

raw_data = np.array(data.raw_ratings, dtype = int)

raw_data[:, 0] -= 1
raw_data[:, 1] -= 1

n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])
shape = (n_users + 1, n_movies + 1)

adj_matrix = np.ndarray(shape, dtype = int)

for user_id, movie_id, rating, time in raw_data:
    adj_matrix[user_id][movie_id] = 1       #영화를 봤는지 안봤는지 체크

# 유클리드 거리를 사용하여 추천
my_id, my_vector = 0, adj_matrix[0]

best_match, best_match_id, best_match_vector = 9999, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        euclidean_dist = np.sqrt(np.sum(np.square(my_vector - user_vector)))
        if euclidean_dist < best_match:
            best_match = euclidean_dist
            best_match_id = user_id
            best_match_vector = user_vector
            
print('euclidean_dist')
print('Best Match : {}, Best Match ID : {}'.format(best_match, best_match_id))

recommend_list = []

for i, log in enumerate(zip(my_vector, best_match_vector)):
    log1, log2 = log
    if log1 < 1. and log2 > 0.:
        recommend_list.append(i)

# 737번은 봤지만 나는 보지 않은 영화 추천 리스트
print(recommend_list)
