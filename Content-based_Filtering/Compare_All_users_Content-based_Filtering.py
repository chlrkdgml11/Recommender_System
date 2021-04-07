import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import matplotlib.pyplot as plt

df_movie = pd.read_csv("./Data/content_movies.csv", sep=',', header=0)
df_rating = pd.read_csv("./Data/content_ratings.csv", sep=',', header=0)
del df_rating['timestamp']
df_rating = df_rating.sort_values(['userId', 'rating'], ascending=[True, False])

count_vector = CountVectorizer(ngram_range=(1, 3))
c_vector_genres = count_vector.fit_transform(df_movie['genres'])
cosine_similarity = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]


def get_recommend_movie_list(df_movie, movie_title, top=30):
    # 특정 영화와 비슷한 영화를 추천해야 하기 때문에 '특정 영화' 정보를 뽑아낸다.
    target_movie_index = df_movie[df_movie['title'] == movie_title].index.values
    
    #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = cosine_similarity[target_movie_index, :top].reshape(-1)
    #본인을 제외
    sim_index = sim_index[sim_index != target_movie_index]

    #data frame으로 만들고 vote_count으로 정렬한 뒤 return
    result = df_movie.iloc[sim_index][:10]
    return result


for i in range(141, 611):
    genres_list  = np.array(['Mystery', 'Thriller', 'Crime', 'Adventure', 'Comedy', 'Romance', 'Action', 'Drama', 'War', 'Sci-Fi', 'Children', 'Western', 'Animation', 'Fantasy', 'Musical', 'Film-Noir', 'Horror', 'Documentary', 'IMAX', '(no genres listed)'])
    genres_count_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    genres_rating_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for m in range(df_movie.shape[0]):
        for p in df_movie.loc[m].genres.split('|'):
            for b in range(20):
                if(genres_list[b] == p):
                    genres_count_list[b] += 1

    if(i % 100 == 0):
        print('i = ', i)

    # 해당 유저가 평점을 매긴 movieId 배열 저장
    np_selected_movies = np.array(df_rating[df_rating["userId"] == i].movieId)

    for j in range(len(np_selected_movies)):
        for k in df_movie[df_movie["movieId"] == np_selected_movies[j]].genres.values[0].split('|'):
            for l in range(20):
                if(genres_list[l] == k):
                    # genres_count_list[l] += 1
                    genres_rating_list[l] += df_rating[(df_rating['userId'] == i) & (df_rating['movieId'] == np_selected_movies[j])].rating.values[0]


    for t in range(0,20):
        if(genres_count_list[t] == 0):
            genres_rating_list[t] = 0
            continue
        genres_rating_list[t] = float(genres_rating_list[t]) / float(genres_count_list[t])

    plt.figure(figsize=(25,10))
    for o in range(19):
        plt.bar(genres_list[o], genres_rating_list[o], label='%d_user' %(i))

    plt.title('Rated Movies')
    plt.xlabel('Genres')
    plt.ylabel('ratings')
    plt.grid()
    plt.savefig('./Content_All_Image/%d' %(i))
    plt.close()

    # ab = np.zeros(genres_list.size, dtype=[('var1', 'U6'), ('var2', float)])
    # ab['var1'] = genres_list
    # ab['var2'] = genres_rating_list

    # np.savetxt('./Content_All_Text/%d.txt' %(i), ab, fmt="%7s %10f")
