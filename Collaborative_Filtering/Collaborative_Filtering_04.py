# 영화명과 rating값을 pearson으로 이용하여 비슷한 영화 찾기
import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# print(ratings.head(10))
# print(movies.head(10))

ratings = pd.merge(movies, ratings).drop(['genres','timestamp'],axis=1)

print(ratings.head())

# index값은 userId로 하고 columns은 title, 값들은 rating으로
userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')

print(userRatings.head())

# 열중에 10개 미만의 rating이 있는 열은 삭제
# 그리고 남은 열의 결측값은 0으로 채운다
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)

print(userRatings)

corrMatrix = userRatings.corr(method='pearson')

print(corrMatrix)

def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings

romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()

for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

print(similar_movies.head())

print(similar_movies.sum().sort_values(ascending=False).head(20))
