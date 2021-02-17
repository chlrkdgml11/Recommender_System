import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('u.data', sep='\t', header=None)

df.columns = ["user_id", "item_id", "rating", "timestamp"]

# print(df.head())

n_users = df.user_id.max()
n_items = df.item_id.max()

# print(n_users, n_items)

ratings = np.zeros((n_users, n_items))

for row in df.itertuples():
    ratings[row[1]-1,row[2]-1] = row[3]

ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)

# print(ratings_train.shape)

ratings_train = pd.DataFrame(ratings_train).T

user_distances = ratings_train.corr(method='pearson')

# print(user_distances)

# user_distances = cosine_similarity(ratings_train)

# print('ratings_train\n', ratings_train)
# print('ratings_train.shape\n', ratings_train.shape)
# print('ratings_test\n', ratings_test)
# print('ratings_test.shape\n', ratings_test.shape)
# print('user_distances\n', user_distances)
# print('user_distances.shape\n', user_distances.shape)

user_distances = pd.DataFrame.to_numpy(user_distances)

print('user_distances\n', user_distances)

print('np.min(user_distances)\n', np.min(user_distances))

print('np.max(user_distances)\n', np.max(user_distances))

# print(user_distances.shape)
# print(ratings_train.shape)

user_pred = user_distances.dot(ratings_train.T) / np.array([np.abs(user_distances).sum(axis=1)]).T

print('user_pred\n', user_pred)

print('np.min(user_pred)\n', np.min(user_pred))

print('np.max(user_pred)\n', np.max(user_pred))

print()
# print(user_pred.shape)

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

ratings_train = pd.DataFrame.to_numpy(ratings_train)

print(user_pred)
print(ratings_train.T)

print(np.sqrt(get_mse(user_pred, ratings_train.T)))

k=5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
neigh.fit(ratings_train)

top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
print('top_k_distances\n', top_k_distances)
print('top_k_distances.shape\n', top_k_distances.shape)
print('top_k_users\n', top_k_users)
print('top_k_users.shape\n', top_k_users.shape)

user_pred_k = np.zeros(ratings_train.shape)

for i in range(ratings_train.shape[0]):
    if(i%50==0):
        print("cnt = ", i)
    user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print('user_pred_k\n', user_pred_k)
print('user_pred_k.shape\n', user_pred_k.shape)

dfCf = []

sum = 0
n = 0
# for i in range(len(user_pred_k)):
#     for j in range(len(user_pred_k[0])):
#         if(user_pred_k[i][j])

avg = sum / n

print('RMSE\n', np.sqrt(get_mse(user_pred_k, ratings_train)))
