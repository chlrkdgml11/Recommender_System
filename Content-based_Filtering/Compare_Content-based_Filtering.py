import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval


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


modified_df = df_rating.loc[(df_rating['rating']  >= 4.5)]

lack_user = []


for i in range(1,611):
    if(i % 50 == 0):
        print('i = ', i)
    if(modified_df[modified_df["userId"] == i].shape[0] < 10):
        lack_user.append(i)
print(lack_user)



# genres_list = [['Mystery', 0], ['Thriller', 0], ['Crime', 0], ['Adventure', 0], ['Comedy', 0], ['Romance', 0], ['Action', 0], ['Drama', 0], ['War', 0], ['Sci-Fi', 0], ['Children', 0], ['Western', 0], ['Animation', 0], ['Fantasy', 0], ['Musical', 0], ['Film-Noir', 0], ['Horror', 0], ['Documentary', 0], ['IMAX', 0], ['(no genres listed)', 0]]



for i in range(1, 611):
    genres_list  = np.array(['Mystery', 'Thriller', 'Crime', 'Adventure', 'Comedy', 'Romance', 'Action', 'Drama', 'War', 'Sci-Fi', 'Children', 'Western', 'Animation', 'Fantasy', 'Musical', 'Film-Noir', 'Horror', 'Documentary', 'IMAX', '(no genres listed)'])
    genres_count_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    if i in lack_user:
        continue

    if(i % 50 == 0):
        print('i = ', i)

    np_selected_movies = np.array(modified_df[modified_df["userId"] == i].movieId)

    for j in range(len(np_selected_movies)):
        if(j >= 10):
            break
        if(np_selected_movies[j] == 2851 or np_selected_movies[j] == 838 or np_selected_movies[j] == 34048):
            continue
        result = get_recommend_movie_list(df_movie, movie_title=df_movie[df_movie["movieId"] == np_selected_movies[j]].title.values[0])

        # print(result.genres.values[0].split('|'))

        for k in result.genres.values[0].split('|'):
            for l in range(20):
                if(genres_list[l] == k):
                    genres_count_list[l] += 1

    ab = np.zeros(genres_list.size, dtype=[('var1', 'U6'), ('var2', float)])
    ab['var1'] = genres_list
    ab['var2'] = genres_count_list

    np.savetxt('./Content_text/%d.txt' %(i), ab, fmt="%7s %10d")
