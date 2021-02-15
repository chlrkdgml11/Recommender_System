import numpy as np
from sklearn.decomposition import randomized_svd, non_negative_factorization
from surprise import Dataset

data = Dataset.load_builtin('ml-100k', prompt = False)

raw_data = np.array(data.raw_ratings, dtype = int)

raw_data[:, 0] -= 1
raw_data[:, 1] -= 1

n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])
shape = (n_users + 1, n_movies + 1)

adj_matrix = np.ndarray(shape, dtype = int)

for user_id, movie_id, rating, time in raw_data:
    adj_matrix[user_id][movie_id] = rating

# 코사인 유사도를 사용하여 추천
def compute_cos_similarity(v1, v2):
    norm1 = np.sqrt(np.sum(np.square(v1)))
    norm2 = np.sqrt(np.sum(np.square(v2)))
    dot = np.dot(v1, v2)
    return dot / (norm1 * norm2)


# 비음수 행렬 분해를 사용한 하이브리드 추천
A, B, iter = non_negative_factorization(adj_matrix, n_components = 2)

np.matmul(A, B)

# 항목 기반 추천
my_id, my_vector = 0, B.T[0]

best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(B.T):
    if my_id != user_id:
        cos_similarity = compute_cos_similarity(my_vector, user_vector)
        if cos_similarity > best_match:
            best_match = cos_similarity
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match : {}, Best Match ID : {}'.format(best_match, best_match_id))

recommend_list = []

for i, user_vector in enumerate(adj_matrix):
    if adj_matrix[i][my_id] > 0.9:
        recommend_list.append(i)

print(recommend_list)
