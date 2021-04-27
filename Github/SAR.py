import sys
sys.path.append("./")

import logging
import numpy as np
import pandas as pd
import scrapbook as sb
from sklearn.preprocessing import minmax_scale

from reco_utils.common.python_utils import binarize
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from reco_utils.recommender.sar import SAR



# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)

n_users = data.userID.max()
n_items = 445

data = data.loc[(data['userID']  <= n_users) & (data['itemID']  <= n_items)]

# Convert the float precision to 32-bit in order to reduce memory consumption 
data['rating'] = data['rating'].astype(np.float32)

train, test = python_stratified_split(data, ratio=0.75, col_user='userID', col_item='itemID', seed=42)


print("""
Train:
Total Ratings: {train_total}
Unique Users: {train_users}
Unique Items: {train_items}

Test:
Total Ratings: {test_total}
Unique Users: {test_users}
Unique Items: {test_items}
""".format(
    train_total=len(train),
    train_users=len(train['userID'].unique()),
    train_items=len(train['itemID'].unique()),
    test_total=len(test),
    test_users=len(test['userID'].unique()),
    test_items=len(test['itemID'].unique()),
))

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard", 
    time_decay_coefficient=30, 
    timedecay_formula=True,
    normalize=True
)


with Timer() as train_time:
    model.fit(train)



print("Took {} seconds for training.".format(train_time.interval))



with Timer() as test_time:
    top_k = model.recommend_k_items(test, remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))

print(top_k)




np_predict = np.array(top_k)
np_test = np.array(test)


ratings_predict = np.zeros((n_users, n_items))
for i in range(np_predict.shape[0]):
    ratings_predict[int(np_predict[i][0])-1][int(np_predict[i][1])-1] = np_predict[i][2]


ratings_test = np.zeros((n_users, n_items))
for i in range(np_test.shape[0]):
    ratings_test[int(np_test[i][0])-1][int(np_test[i][1])-1] = np_test[i][2]










sum = 0
cnt = 0
for i in range(n_users):
    for j in range(n_items):
        if(ratings_predict[i][j] != 0 and ratings_test[i][j] != 0):
            sum += (ratings_predict[i][j] - ratings_test[i][j]) ** 2
            cnt += 1

print(sum / cnt)



