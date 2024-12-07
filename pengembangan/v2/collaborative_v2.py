import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score

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

user_counts = recommendations_df['user_id'].value_counts()
print("list id user : ",user_counts[user_counts >= 7].index)

# Informasi dasar tentang dataset
print("Informasi Dasar tentang Dataset Games:")
print(games_df.info())
print("\nInformasi Dasar tentang Dataset Recommendations:")
print(recommendations_df.info())

# Mengecek nilai yang hilang di dataset
print("\nNilai yang Hilang di Dataset Games:")
print(games_df.isnull().sum())
print("\nNilai yang Hilang di Dataset Recommendations:")
print(recommendations_df.isnull().sum())

# Statistik dasar dari dataset
print("\nStatistik Dataset Games:")
print(games_df.describe())

# Visualisasi menggunakan seaborn
sns.set(style="whitegrid")

# 1. Tren Rilis Game
games_df['release_year'] = pd.to_datetime(games_df['release_date']).dt.year  # Mengambil tahun rilis
release_trend = games_df['release_year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x=release_trend.index, y=release_trend.values, marker="o")
plt.title("Tren Rilis Game")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Game yang Dirilis")
plt.show()

# 2. Distribusi 10 Genre Teratas
genre_counts = games_df['genres'].str.split(";").explode().value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.title("Distribusi 10 Genre Teratas")
plt.xlabel("Genre")
plt.ylabel("Jumlah")
plt.xticks(rotation=45)
plt.show()

# 3. Distribusi Rating
games_df['positive_ratio'] = games_df['positive_ratings'] / (
    games_df['positive_ratings'] + games_df['negative_ratings'])  # Rasio positif
plt.figure(figsize=(12, 6))
sns.histplot(games_df['positive_ratio'], bins=50, kde=True, color="blue")
plt.title("Distribusi Rasio Positif")
plt.xlabel("Rasio Positif")
plt.ylabel("Jumlah")
plt.show()

# 4. Analisis Harga Game
plt.figure(figsize=(15, 6))
sns.histplot(data=games_df[games_df['price'] < 100], x='price', bins=50, color="blue")
plt.title('Distribusi Harga Game (< $100)')
plt.xlabel('Harga ($)')
plt.ylabel('Jumlah')
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
item_matrix = svd.components_

# Fungsi untuk mendapatkan user serupa menggunakan Matrix Factorization
def get_similar_users(user_id, user_matrix, n_neighbors=6):
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

# Contoh penggunaan
user_id_example = 1288609  # Masukkan user_id yang valid dari dataset Anda
recommended_games = recommend_games(user_id_example, n_neighbors=6)

# Menampilkan hasil rekomendasi
print(f"Game yang direkomendasikan untuk user {user_id_example} adalah:")
for game in recommended_games:
    print(f"{game['title']} | Rasio Positif: {game['positive_ratings']} | Genre: {game['genres']} | Harga: {game['price']}")

# # Fungsi untuk menghitung Precision@k, Recall@k, F1@k
# def precision_recall_f1_at_k(y_true, y_pred, k=5):
#     """
#     Menghitung Precision@k, Recall@k, F1@k.
#     y_true: nilai sebenarnya (1 untuk direkomendasikan, 0 untuk tidak)
#     y_pred: hasil prediksi model
#     k: jumlah rekomendasi yang diambil
#     """
#     precision_list = []
#     recall_list = []
#     f1_list = []

#     for true, pred in zip(y_true, y_pred):
#         # Ambil top k prediksi
#         top_k_pred = pred[:k]
#         top_k_true = true[:k]

#         # Hitung Precision, Recall, dan F1 untuk top k
#         precision = precision_score(top_k_true, top_k_pred, average='binary', zero_division=0)
#         recall = recall_score(top_k_true, top_k_pred, average='binary', zero_division=0)
#         f1 = f1_score(top_k_true, top_k_pred, average='binary', zero_division=0)

#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     # Rata-rata Precision, Recall, dan F1
#     precision_at_k = np.mean(precision_list)
#     recall_at_k = np.mean(recall_list)
#     f1_at_k = np.mean(f1_list)

#     return precision_at_k, recall_at_k, f1_at_k

# # Fungsi untuk menghitung RMSE dan MAE
# def evaluate_model(user_game_matrix, recommendations_df, games_df, n_neighbors=6):
#     # Prediksi hasil (apakah game akan direkomendasikan)
#     y_true = []
#     y_pred = []

#     for user_id in recommendations_df['user_id'].unique():
#         # Ambil rekomendasi untuk setiap pengguna
#         recommended_games = recommend_games(user_id, n_neighbors)
#         recommended_game_ids = [game['title'] for game in recommended_games]
        
#         # Ambil data asli untuk membandingkan apakah game direkomendasikan
#         user_games = recommendations_df[recommendations_df['user_id'] == user_id]
#         actual_recommendations = user_games['is_recommended'].values  # 1 atau 0

#         # Buat vektor untuk hasil yang diprediksi dan yang sebenarnya
#         pred_vector = [1 if game in recommended_game_ids else 0 for game in games_df['title']]
#         true_vector = [1 if game in user_games['app_id'].values else 0 for game in games_df['app_id']]

#         y_true.append(true_vector)
#         y_pred.append(pred_vector)

#     # Menghitung RMSE dan MAE
#     rmse = np.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred)))
#     mae = mean_absolute_error(np.array(y_true), np.array(y_pred))

#     # Menghitung Precision@k, Recall@k, F1@k
#     precision_at_k, recall_at_k, f1_at_k = precision_recall_f1_at_k(y_true, y_pred, k=5)

#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"Precision@5: {precision_at_k:.4f}")
#     print(f"Recall@5: {recall_at_k:.4f}")
#     print(f"F1@5: {f1_at_k:.4f}")

# # Evaluasi model
# evaluate_model(user_game_matrix, recommendations_df, games_df, n_neighbors=6)