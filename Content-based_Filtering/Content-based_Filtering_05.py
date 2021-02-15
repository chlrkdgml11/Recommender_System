import numpy as np
from surprise import Dataset

data = Dataset.load_builtin('ml-100k', prompt=False)

raw_data = np.array(data.raw_ratings, dtype = int)

raw_data[:, 0] -= 1
raw_data[:, 1] -= 1

n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])
shape = (n_users + 1, n_movies + 1)

# 기존 방법에 명시적 피드백(사용자의 영화 점수) 추가
adj_matrix = np.ndarray(shape, dtype = int)
for user_id, movie_id, rating, time in raw_data:
    adj_matrix[user_id][movie_id] = rating      # 기존은 1로 봤는지 보지 않았는지 체크했었음

def compute_cos_similarity(v1, v2):
    norm1 = np.sqrt(np.sum(np.square(v1)))
    norm2 = np.sqrt(np.sum(np.square(v2)))
    dot = np.dot(v1, v2)
    return dot / (norm1 * norm2)

# 코사인 유사도를 사용하여 추천
my_id, my_vector = 0, adj_matrix[0]

best_match, best_match_id, best_match_vector = -1, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        cos_similarity = compute_cos_similarity(my_vector, user_vector)
        if cos_similarity > best_match:
            best_match = cos_similarity
            best_match_id = user_id
            best_match_vector = user_vector

print('Best Match : {}, Best Match ID : {}'.format(best_match, best_match_id))

recommend_list = []

for i, log in enumerate(zip(my_vector, best_match_vector)):
    log1, log2 = log
    if log1 < 1. and log2 > 0.:
        recommend_list.append(i)

# 915번은 봤지만 나는 보지 않은 영화 추천 리스트
print(recommend_list)
