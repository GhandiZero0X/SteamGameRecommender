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

# Prepare improved recommendation data with proper sample size
def prepare_improved_recommendation_data(user_df, game_df, recommendation_df, sample_size=50000):
    """Prepare the recommendation data with better filtering"""
    print("Sampling and preparing data...")

    # Adjust sample_size to be the minimum of the requested size and the available size
    sample_size = min(sample_size, len(recommendation_df))

    # Sample recommendations
    recommendation_sample = recommendation_df.sample(n=sample_size, random_state=0)
    print(f"Initial sample size: {len(recommendation_sample)}")
    
    # Convert to integer and add positive/negative weight
    recommendation_sample['is_recommended'] = recommendation_sample['is_recommended'].astype(int)
    
    # Filter users and games
    user_interactions = recommendation_sample['user_id'].value_counts()
    game_interactions = recommendation_sample['app_id'].value_counts()
    
    min_user_interactions = 2
    min_game_interactions = 2
    
    valid_users = user_interactions[user_interactions >= min_user_interactions].index
    valid_games = game_interactions[game_interactions >= min_game_interactions].index
    
    filtered_recommendations = recommendation_sample[
        (recommendation_sample['user_id'].isin(valid_users)) & 
        (recommendation_sample['app_id'].isin(valid_games))
    ]
    
    print(f"Filtered recommendations: {len(filtered_recommendations)}")
    print(f"Unique users: {len(valid_users)}")
    print(f"Unique games: {len(valid_games)}")
    
    # Create interaction matrix
    print("Creating interaction matrix...")
    interactions = filtered_recommendations.pivot(
        index='user_id',
        columns='app_id',
        values='is_recommended'
    ).fillna(0)
    
    return interactions.astype(np.int8)

# Display recommendations
def display_recommendations(user_id, recommender, interactions):
    """Display recommendations with improved formatting and error handling"""
    try:
        user_idx = interactions.index.get_loc(user_id)
        recommendations = recommender.get_recommendations(user_idx)
        
        if not recommendations:
            return "No recommendations found for this user."
        
        df = pd.DataFrame(recommendations)
        df = df[['title', 'rating', 'price']]
        df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
        return df
    
    except KeyError:
        return "User not found in the sample dataset"
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# Create interaction matrix
print("Preparing recommendation data...")
interactions = prepare_improved_recommendation_data(user_df, game_df, recommendation_df)
interactions_sparse = csr_matrix(interactions.values)

# Test recommendations
print("\nTesting recommendations...")
sample_user_ids = list(interactions.index[:5])
for user_id in sample_user_ids:
    print(f"\nRecommendations for user {user_id}:")
    recommendations_df = display_recommendations(user_id, recommender, interactions)
    print(recommendations_df)

# Print system statistics
print("\nSystem Statistics:")
print(f"Number of users in sample: {interactions.shape[0]}")
print(f"Number of games in sample: {interactions.shape[1]}")
