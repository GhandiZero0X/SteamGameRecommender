import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

# Fungsi untuk mengurangi penggunaan memori DataFrame
def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# Memuat data
games_df = reduce_memory(pd.read_csv('games.csv'))  # Data game
recommendations_df = reduce_memory(pd.read_csv('recommendations.csv'))  # Data rekomendasi

# 1. Visualisasi Tren Rilis Game
games_df['release_year'] = pd.to_datetime(games_df['release_date']).dt.year  # Mengambil tahun rilis
release_trend = games_df['release_year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x=release_trend.index, y=release_trend.values, marker="o")
plt.title("Tren Rilis Game")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Game yang Dirilis")
plt.grid()
plt.show()

# 2. Visualisasi Distribusi Genre Game
genre_counts = games_df['genres'].str.split(";").explode().value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.title("Distribusi 10 Genre Teratas")
plt.xlabel("Genre")
plt.ylabel("Jumlah")
plt.xticks(rotation=45)
plt.show()

# 3. Visualisasi Distribusi Rasio Positif
games_df['positive_ratio'] = games_df['positive_ratings'] / (
    games_df['positive_ratings'] + games_df['negative_ratings'])  # Rasio positif
plt.figure(figsize=(12, 6))
sns.histplot(games_df['positive_ratio'], bins=50, kde=True, color="blue")
plt.title("Distribusi Rasio Positif")
plt.xlabel("Rasio Positif (%)")
plt.ylabel("Jumlah")
plt.grid()
plt.show()

# 4. Visualisasi Distribusi Harga Game
plt.figure(figsize=(15, 6))
sns.histplot(data=games_df[games_df['price'] < 100], x='price', bins=50, color="blue")
plt.title("Distribusi Harga Game (< $100)")
plt.xlabel("Harga ($)")
plt.ylabel("Jumlah")
plt.grid()
plt.show()

# Mapping setiap user dan item ke nilai numerik unik
user_ids = recommendations_df['user_id'].astype('category').cat.codes
item_ids = recommendations_df['app_id'].astype('category').cat.codes

# Mendapatkan ID user dan game unik
unique_user_ids = recommendations_df['user_id'].astype('category').cat.categories
unique_item_ids = recommendations_df['app_id'].astype('category').cat.categories

# Menggunakan 'is_recommended' sebagai preferensi (1 untuk direkomendasikan, 0 untuk tidak)
user_game_matrix = coo_matrix((recommendations_df['is_recommended'], (user_ids, item_ids)))

# Melatih model Matrix Factorization menggunakan SVD (Singular Value Decomposition)
svd = TruncatedSVD(n_components=50)
user_matrix = svd.fit_transform(user_game_matrix)
print("\nUser Matrix :", user_matrix)
item_matrix = svd.components_

# Fungsi untuk mendapatkan user serupa menggunakan Matrix Factorization
# fungsi ini digunakan untuk Menghitung kemiripan kosinus antara pengguna target dan pengguna lain, 
# lalu mengambil pengguna dengan kemiripan tertinggi.
def get_similar_users(user_id, user_matrix, n_neighbors=6):
    if user_id not in unique_user_ids:
        return []
    user_index = np.where(unique_user_ids == user_id)[0][0]  # Index user
    user_vector = user_matrix[user_index].reshape(1, -1)
    cosine_similarities = cosine_similarity(user_vector, user_matrix)
    similar_users = cosine_similarities.argsort()[0][-n_neighbors-1:-1]  # Top N user serupa
    return [unique_user_ids[i] for i in similar_users]

# Fungsi untuk mendapatkan game yang direkomendasikan berdasarkan user serupa
def recommend_games(user_id, n_neighbors=6):
    similar_users = get_similar_users(user_id, user_matrix, n_neighbors=n_neighbors)
    recommended_games = []
    for user in similar_users:
        user_games = recommendations_df[recommendations_df['user_id'] == user]['app_id'].unique()
        for game_id in user_games:
            game_info = games_df[games_df['app_id'] == game_id][['title', 'positive_ratings', 'genres', 'price']]
            recommended_games.extend(game_info.to_dict(orient='records'))
    return list({game['title']: game for game in recommended_games}.values())  # Menghapus duplikat berdasarkan judul game

# Fungsi utama untuk digunakan di Flask
def get_recommendations_for_user(user_id, n_neighbors=6):
    try:
        user_id = int(user_id)  # Pastikan user_id berupa integer
    except ValueError:
        return []  # Jika input tidak valid, kembalikan daftar kosong

    if user_id not in unique_user_ids:
        return []  # Jika user_id tidak ditemukan, kembalikan daftar kosong

    # Dapatkan rekomendasi
    recommendations = recommend_games(user_id, n_neighbors=n_neighbors)
    return recommendations