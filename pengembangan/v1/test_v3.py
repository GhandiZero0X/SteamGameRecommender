# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')  # Changed to 'ggplot' to avoid error with 'seaborn'

# Load data
user_df = pd.read_csv("users.csv")
game_df = pd.read_csv("games.csv")
recommendation_df = pd.read_csv("recommendations.csv")

# Basic information about datasets
print("Users Dataset Info:")
print(user_df.info())
print("\nGames Dataset Info:")
print(game_df.info())
print("\nRecommendations Dataset Info:")
print(recommendation_df.info())

# Check for missing values
print("\nMissing Values:")
print("Users Dataset:\n", user_df.isnull().sum())
print("\nGames Dataset:\n", game_df.isnull().sum())
print("\nRecommendations Dataset:\n", recommendation_df.isnull().sum())

# Convert date_release to datetime
game_df['date_release'] = pd.to_datetime(game_df['date_release'])
recommendation_df['date'] = pd.to_datetime(recommendation_df['date'])

# Basic statistics
print("\nGames Dataset Statistics:")
print(game_df.describe())

# 1. Game Release Trends
plt.figure(figsize=(15, 6))
game_df['date_release'].dt.year.value_counts().sort_index().plot(kind='line')
plt.title('Number of Games Released per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.show()

# 2. Platform Distribution
platforms = pd.DataFrame({
    'Windows': game_df['win'].sum(),
    'Mac': game_df['mac'].sum(),
    'Linux': game_df['linux'].sum(),
    'Steam Deck': game_df['steam_deck'].sum()
}, index=[0]).T
plt.figure(figsize=(10, 6))
platforms.plot(kind='bar')
plt.title('Games Available by Platform')
plt.ylabel('Number of Games')
plt.show()

# 3. Rating Distribution
plt.figure(figsize=(12, 6))
game_df['rating'].value_counts().plot(kind='bar')
plt.title('Distribution of Game Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 4. Price Analysis
plt.figure(figsize=(15, 6))
sns.histplot(data=game_df[game_df['price_final'] < 100], x='price_final', bins=50)
plt.title('Distribution of Game Prices (< $100)')
plt.xlabel('Price ($)')
plt.show()

# 5. User Activity Analysis
plt.figure(figsize=(12, 6))
sns.histplot(data=user_df[user_df['products'] < 1000], x='products', bins=50)
plt.title('Distribution of Number of Products per User (< 1000 products)')
plt.xlabel('Number of Products')
plt.show()

# Prepare recommendation data
def prepare_recommendation_data(user_df, game_df, recommendation_df, sample_size=10000):
    # Sample the recommendation data
    recommendation_sample = recommendation_df.sample(n=sample_size, random_state=0)
    
    # Convert boolean recommendations to integers (0 and 1)
    recommendation_sample['is_recommended'] = recommendation_sample['is_recommended'].astype(int)
    
    # Create user-game interaction matrix
    interactions = recommendation_sample.pivot(
        index='user_id',
        columns='app_id',
        values='is_recommended'
    ).fillna(0)
    
    # Ensure all values are integers
    interactions = interactions.astype(np.int8)
    
    return interactions

# Create interaction matrix
interactions = prepare_recommendation_data(user_df, game_df, recommendation_df)

# Convert to sparse matrix for efficiency
from scipy.sparse import csr_matrix
interactions_sparse = csr_matrix(interactions.values)

# Print some information about the sparse matrix
print("Sparse matrix shape:", interactions_sparse.shape)
print("Number of non-zero elements:", interactions_sparse.nnz)
print("Sparsity: {:.4f}%".format(100 * interactions_sparse.nnz / (interactions_sparse.shape[0] * interactions_sparse.shape[1])))

# Collaborative filtering with NearestNeighbors
from sklearn.neighbors import NearestNeighbors

class GameRecommender:
    def __init__(self, interactions_matrix, game_df):
        self.interactions_matrix = interactions_matrix
        self.game_df = game_df
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(interactions_matrix)
    
    def get_recommendations(self, user_idx, n_recommendations=5):
        distances, indices = self.model.kneighbors(
            self.interactions_matrix[user_idx].reshape(1, -1),
            n_neighbors=n_recommendations+1
        )
        
        similar_users = indices.flatten()[1:]
        recommended_games = []
        
        for user in similar_users:
            user_games = self.interactions_matrix[user].nonzero()[1]
            for game in user_games:
                if self.interactions_matrix[user_idx, game] == 0:  # User hasn't interacted with this game
                    game_info = self.game_df[self.game_df.index == game].iloc[0]
                    recommended_games.append({
                        'title': game_info['title'],
                        'rating': game_info['rating'],
                        'price': game_info['price_final']
                    })
        
        return recommended_games[:n_recommendations]

# Initialize recommender
recommender = GameRecommender(interactions_sparse, game_df)

# Display user IDs
print("\nUser IDs in the interaction matrix:")
print(interactions.index.tolist())  # This will print all user IDs in the interaction matrix

# Input manual data
print("\nInputting manual data for recommendations...")

# Define a function for manual input
def get_user_input():
    try:
        user_id = int(input("Enter user ID: "))
        app_id = int(input("Enter app ID: "))
        is_recommended = int(input("Is the game recommended? (1 for yes, 0 for no): "))
        return user_id, app_id, is_recommended
    except ValueError:
        print("Invalid input. Please enter integers for user ID, app ID, and recommendation status.")
        return None

# Allow user to input multiple recommendations
manual_data = []
while True:
    user_input = get_user_input()
    if user_input:
        manual_data.append(user_input)
    more_data = input("Do you want to input another recommendation? (yes/no): ")
    if more_data.lower() != 'yes':
        break

# Create a DataFrame for the manual input
manual_df = pd.DataFrame(manual_data, columns=['user_id', 'app_id', 'is_recommended'])

# Add the manual data to the recommendation dataframe
recommendation_df = pd.concat([recommendation_df, manual_df], ignore_index=True)

# Re-create interaction matrix with updated data
interactions = prepare_recommendation_data(user_df, game_df, recommendation_df)

# Convert to sparse matrix for efficiency
interactions_sparse = csr_matrix(interactions.values)

# Test recommendations after manual input
print("\nTesting recommendations after manual input...")
sample_user_ids = list(interactions.index[:5])
for user_id in sample_user_ids:
    print(f"\nRecommendations for user {user_id}:")
    recommendations_df = display_recommendations(user_id, recommender, interactions)
    print(recommendations_df)

# Print system statistics
print("\nSystem Statistics:")
print(f"Number of users in sample: {interactions.shape[0]}")
print(f"Number of games in sample: {interactions.shape[1]}")
