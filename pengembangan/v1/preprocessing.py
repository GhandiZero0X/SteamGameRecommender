import pandas as pd
import numpy as np

# Load the two CSV files to inspect their contents
games_path = 'games_description.csv'
recommendations_path = 'steam_game_reviews.csv'

# Read the datasets
games_df = pd.read_csv(games_path)
recommendations_df = pd.read_csv(recommendations_path)

# Display the first few rows and summary information of each dataset
games_summary = games_df.info(), games_df.head()
recommendations_summary = recommendations_df.info(), recommendations_df.head()

# cek missing values
cek_games = games_df.isnull().sum()
cek_rekomendasi = recommendations_df.isnull().sum()
print(cek_games)
print(cek_rekomendasi)

# penanganan missing values
games_df = games_df.dropna()
recommendations_df = recommendations_df.dropna()

# simpan ke file csv
games_df.to_csv('games_description_clean.csv', index=False)
recommendations_df.to_csv('steam_game_reviews_clean.csv', index=False)