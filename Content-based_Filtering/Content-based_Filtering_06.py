import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('./Data/tmdb_5000_movies.csv')
data = data[['id','genres', 'vote_average', 'vote_count','title',  'keywords']]

# print(data)

# 여기서부터의 과정은 rating 불공정을 처리하기 위한 과정
# 투표수의 상위 89%는 약 1683
# 투표수가 1683개 이상인 영화수는 529개
tmp_m = data['vote_count'].quantile(0.89)
tmp_data = data.copy().loc[data['vote_count'] >= tmp_m]

# 투표수의 상위 89%는 약 1838
# 투표수가 1838 이상인 영화수는 481개
m = data['vote_count'].quantile(0.9)
data = data.loc[data['vote_count'] >= m]

C = data['vote_average'].mean()

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    
    return ( v / (v+m) * R ) + (m / (v + m) * C)

# R : 개별 영화 평점
# v : 개별 영화에 평점을 투표한 횟수
# m : 250위 안에 들어야 하는 최소 투표 (여기서는 500위라고 가정)
# c : 전체 영화에 대한 평균 평점

# data에 score 열 추가
data['score'] = data.apply(weighted_rating, axis = 1)

# genres와 keywords의 type 변환
data['genres'] = data['genres'].apply(literal_eval)
data['keywords'] = data['keywords'].apply(literal_eval)
data['genres'] = data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))
data['keywords'] = data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

# 데이터 저장
data.to_csv('./pre_tmdb_5000_movies.csv', index = False)


# 본격적인 Content based filtering 시작
# 장르 단어를 벡터화 시켜서 저장
count_vector = CountVectorizer(ngram_range=(1, 3))
c_vector_genres = count_vector.fit_transform(data['genres'])

# print(c_vector_genres)


#코사인 유사도를 구한 벡터를 미리 저장
cosine_similarity = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]
# print(cosine_similarity)
# print('cosine_similarity(c_vector_genres, c_vector_genres)\n', cosine_similarity(c_vector_genres, c_vector_genres))
# print('cosine_similarity\n', cosine_similarity)


def get_recommend_movie_list(df, movie_title, top=30):
    # 특정 영화와 비슷한 영화를 추천해야 하기 때문에 '특정 영화' 정보를 뽑아낸다.
    target_movie_index = df[df['title'] == movie_title].index.values
    print(target_movie_index)
    # print('target_movie_index\n', df[df['title'] == movie_title].index.values)
    
    #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = cosine_similarity[target_movie_index, :top].reshape(-1)
    #본인을 제외
    sim_index = sim_index[sim_index != target_movie_index]
    # print(sim_index)

    #data frame으로 만들고 vote_count으로 정렬한 뒤 return
    result = df.iloc[sim_index].sort_values('score', ascending=False)[:10]
    # print(result)
    return result

result = get_recommend_movie_list(data, movie_title='The Dark Knight Rises')
# print(result)

# print("result.iloc[0]['genres']\n", result.iloc[0])

# print("result.iloc[1]['genres']\n", result.iloc[1])

genres_array = result.iloc[0]['genres'].split(' ')

print(genres_array)
