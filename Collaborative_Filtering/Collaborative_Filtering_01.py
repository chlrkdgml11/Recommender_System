from surprise import KNNBasic, SVD, SVDpp, NMF
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k', prompt = False)

print(data)
print(type(data))

# KNN(최근접 이웃 알고리즘)을 사용한 협업 필터링
model = KNNBasic()
cross_validate(model, data, measures = ['rmse', 'mae'], cv = 5, n_jobs = 4, verbose = True)


# SVD(특이값 분해)을 사용한 협업 필터링
model = SVD()
cross_validate(model, data, measures = ['rmse', 'mae'], cv = 5, n_jobs = 4, verbose = True)


# NMF(음수 미포함 행렬 분해)을 사용한 협업 필터링
model = NMF()
cross_validate(model, data, measures = ['rmse', 'mae'], cv = 5, n_jobs = 4, verbose = True)


# SVDpp을 사용한 협업 필터링(오랜 시간이 걸림)
# model = SVDpp()
# cross_validate(model, data, measures = ['rmse', 'mae'], cv = 5, n_jobs = 4, verbose = True)
