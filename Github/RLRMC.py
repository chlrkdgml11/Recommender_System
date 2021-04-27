import numpy as np
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append("./")
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.dataset import movielens
from reco_utils.recommender.rlrmc.RLRMCdataset import RLRMCdataset 
from reco_utils.recommender.rlrmc.RLRMCalgorithm import RLRMCalgorithm 
# Pymanopt installation is required via
# pip install pymanopt 
from reco_utils.evaluation.python_evaluation import (
    rmse, mae
)



# print("Pandas version: {}".format(pd.__version__))
# print("System version: {}".format(sys.version))

# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters

# rank of the model, a positive integer (usually small), required parameter
rank_parameter = 10
# regularization parameter multiplied to loss function, a positive number (usually small), required parameter
regularization_parameter = 0.001
# initialization option for the model, 'svd' employs singular value decomposition, optional parameter
initialization_flag = 'svd' #default is 'random'
# maximum number of iterations for the solver, a positive integer, optional parameter
maximum_iteration = 100 #optional, default is 100
# maximum time in seconds for the solver, a positive integer, optional parameter
maximum_time = 300#optional, default is 1000

# Verbosity of the intermediate results
verbosity=0 #optional parameter, valid values are 0,1,2, default is 0
# Whether to compute per iteration train RMSE (and test RMSE, if test data is given)
compute_iter_rmse=True #optional parameter, boolean value, default is False

df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)

n_users = df.userID.max()
n_items = 445

df = df.loc[(df['userID']  <= n_users) & (df['itemID']  <= n_items)]

## If both validation and test sets are required
# train, validation, test = python_random_split(df,[0.6, 0.2, 0.2])

## If validation set is not required
train, test = python_random_split(df,[0.8, 0.2])

print('train shape\n', train.shape)
print('test shape\n', test.shape)

## If test set is not required
# train, validation = python_random_split(df,[0.8, 0.2])

## If both validation and test sets are not required (i.e., the complete dataset is for training the model)
# train = df


# data = RLRMCdataset(train=train, validation=validation, test=test)
data = RLRMCdataset(train=train, test=test) # No validation set
# data = RLRMCdataset(train=train, validation=validation) # No test set
# data = RLRMCdataset(train=train) # No validation or test set

model = RLRMCalgorithm(rank = rank_parameter,
                       C = regularization_parameter,
                       model_param = data.model_param,
                       initialize_flag = initialization_flag,
                       maxiter=maximum_iteration,
                       max_time=maximum_time)

# start_time = time.time()

model.fit(data,verbosity=verbosity)

# fit_and_evaluate will compute RMSE on the validation set (if given) at every iteration
# model.fit_and_evaluate(data,verbosity=verbosity)

# train_time = time.time() - start_time # train_time includes both model initialization and model training time. 

# print("Took {} seconds for training.".format(train_time))



## Obtain predictions on (userID,itemID) pairs (60586,54775) and (52681,36519) in Movielens 10m dataset
# output = model.predict([60586,52681],[54775,36519]) # Movielens 10m dataset

# Obtain prediction on the full test set
predictions_ndarr = model.predict(test['userID'].values,test['itemID'].values)


predictions_df = pd.DataFrame(data={"userID": test['userID'].values, "itemID":test['itemID'].values, "prediction":predictions_ndarr})

## Compute test RMSE 
eval_rmse = rmse(test, predictions_df)



print('test\n', test)
print('predict\n', predictions_df)


np_predict = np.array(predictions_df)
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