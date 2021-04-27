import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cornac
import papermill as pm
import scrapbook as sb

import sys
sys.path.append("./")
from reco_utils.recommender.cornac.cornac_utils import predict_ranking

array_01 = [[0 for col in range(2)] for row in range(51)]
array_02 = [[0 for col in range(2)] for row in range(51)]


df = pd.read_csv("./Data/u.data", sep='\t', header=None)
df.columns = ["user_id", "item_id", "rating", 'timestamp']
del df["timestamp"]




n_users = df.user_id.max()
n_items = 445

modified_df = df.loc[(df['user_id']  <= n_users) & (df['item_id']  <= n_items)]
df_train, df_test = train_test_split(modified_df, test_size=0.2, random_state=42)

print(df_train)
print(df_test)

np_test = np.array(df_test)
ratings_test = np.zeros((n_users, n_items))
for i in range(np_test.shape[0]):
    ratings_test[int(np_test[i][0])-1][int(np_test[i][1])-1] = np_test[i][2]


train_set = cornac.data.Dataset.from_uir(df_train.itertuples(index=False), seed=42)

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))


NUM_FACTORS = 200
NUM_EPOCHS = 100

bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=42
)

bpr.fit(train_set)
all_predictions = predict_ranking(bpr, df_train, usercol='user_id', itemcol='item_id', remove_seen=True)

print(all_predictions)

np_predict = np.array(all_predictions)

ratings_predict = np.zeros((n_users, n_items))
for i in range(np_predict.shape[0]):
    ratings_predict[int(np_predict[i][0])-1][int(np_predict[i][1])-1] = np_predict[i][2]

print(ratings_predict.shape)



for i in range(n_users):
    for j in range(n_items):
        if(ratings_predict[i][j] < 0):
            ratings_predict[i][j] = 0

sum = 0
cnt = 0

for i in range(n_users):
    for j in range(n_items):
        if(ratings_predict[i][j] != 0 and ratings_test[i][j] != 0):
            sum += (ratings_predict[i][j] - ratings_test[i][j]) ** 2
            cnt += 1


print(ratings_predict)
print('rmse', sum/cnt)

