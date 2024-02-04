import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

user_ratings_df = pd.read_csv("./data/ratings.csv")

movie = pd.read_csv("./data/movies.csv")
movie = movie[['title', 'genres']]

movie_data = pd.concat([user_ratings_df, movie], axis=1)

user_item_matrix_sparse = csr_matrix((movie_data['rating'], (movie_data['userId'], movie_data['movieId'])))

cf_knn_model= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
cf_knn_model.fit(user_item_matrix_sparse)

def movie_recommender_engine(movie_name, matrix, cf_model, n_recs):
    cf_knn_model.fit(matrix)

    movie_id = process.extractOne(movie_name, movie_data['title'])[2]

    distances, indices = cf_model.kneighbors(matrix[movie_id], n_neighbors=n_recs)
    movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    cf_recs = []
    for i in movie_rec_ids:
        cf_recs.append({'Title': movie_data['title'][i[0]], 'Distance': i[1]})

    df = pd.DataFrame(cf_recs, index=range(1, n_recs))

    return df

n_recs = 10
movie_recommender_engine('Frozen', user_item_matrix_sparse, cf_knn_model, n_recs)