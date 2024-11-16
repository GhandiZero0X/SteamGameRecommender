# Import Library
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 1. Load Data
reviews_df = pd.read_csv('steam_game_reviews_clean.csv')
games_df = pd.read_csv('games_description_clean.csv')

# 2. Preprocessing Reviews
# Convert 'hours_played' to numerical ratings (simulated ratings)
reviews_df['hours_played'] = pd.to_numeric(reviews_df['hours_played'], errors='coerce')
reviews_df = reviews_df.dropna(subset=['hours_played'])  # Drop rows with non-numeric hours
reviews_df = reviews_df[reviews_df['hours_played'] > 0]  # Remove rows with zero or negative hours

# Normalize 'hours_played' to a scale of 1-5
reviews_df['simulated_rating'] = reviews_df['hours_played'].apply(lambda x: min(x / 10, 5))  # Normalize to a scale of 1-5

# 3. Filter Necessary Columns for the Recommendation System
ratings = reviews_df[['username', 'game_name', 'simulated_rating']].dropna()

# 4. Address Data Sparsity (Additional Filtering for Popular Games and Active Users)
# Keep users who have rated more than 5 games and games with more than 10 reviews
popular_games = ratings['game_name'].value_counts()
active_users = ratings['username'].value_counts()

ratings = ratings[ratings['game_name'].isin(popular_games[popular_games > 10].index)]
ratings = ratings[ratings['username'].isin(active_users[active_users > 5].index)]

# 5. Prepare Data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['username', 'game_name', 'simulated_rating']], reader)

# 6. Train-Test Split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 7. Train Model with SVD (For Sparse Data)
model = SVD()
model.fit(trainset)

# 8. Evaluate Model
predictions = model.test(testset)
mae = accuracy.mae(predictions)
rmse = accuracy.rmse(predictions)
print(f"Evaluation Results: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

# 9. Generate Recommendations for a User
def collaborative_recommendation_with_details(user_id, model, data, games_df, top_n=10):
    """Generate top N game recommendations for a user and include game details."""
    trainset = data.build_full_trainset()
    all_games = trainset.all_items()
    all_games = [trainset.to_raw_iid(game) for game in all_games]
    
    # Predict scores for all games the user hasn't rated yet
    predictions = [
        (game, model.predict(user_id, game).est) for game in all_games
    ]
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Add details from games_description.csv
    detailed_recommendations = []
    for game, score in recommendations:
        game_details = games_df[games_df['name'] == game]
        if not game_details.empty:
            genre = game_details['genres'].values[0]
            detailed_recommendations.append((game, score, genre))
        else:
            detailed_recommendations.append((game, score, "Unknown"))
    
    return detailed_recommendations

# Example usage
# user_id = 'Sentinowl 224 products in account'  # Replace with a valid user ID
user_id = '011 products in account'  # Convert to Surprise user ID
detailed_recommendations = collaborative_recommendation_with_details(user_id, model, data, games_df, top_n=10)

print("\nCollaborative Filtering Recommendations with Details:")
for game, score_rating, genre in detailed_recommendations:
    print(f"\n{game}: {score_rating:.2f} \nGenre: {genre}")
