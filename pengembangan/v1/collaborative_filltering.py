import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
game_df = pd.read_csv('games.csv')
user_df = pd.read_csv('user_sampled.csv')
recommendation_df = pd.read_csv('recommendation_sampled.csv')

# Merge datasets to include titles
recommendation_df = recommendation_df.merge(game_df[['app_id', 'title']], on='app_id')

# Create user-game matrix using 'is_recommended' as values
user_game_matrix = recommendation_df.pivot_table(
    index='user_id', columns='title', values='is_recommended', fill_value=0
)

# Compute cosine similarity (game-based collaborative filtering)
game_similarity = cosine_similarity(user_game_matrix.T)
game_similarity_df = pd.DataFrame(game_similarity, index=user_game_matrix.columns, columns=user_game_matrix.columns)

# Function to get game recommendations
def recommend_games(input_games, game_similarity_df, top_n=5):
    """
    Recommend games based on input games.
    :param input_games: List of game titles (e.g., ["Among Us", "PUBG", ...])
    :param game_similarity_df: DataFrame of game similarity scores
    :param top_n: Number of recommendations to return
    :return: List of recommended games
    """
    scores = pd.Series(dtype=float)
    
    for game in input_games:
        if game in game_similarity_df.index:
            # Add similarity scores for the input game
            scores = scores.add(game_similarity_df[game], fill_value=0)
        else:
            print(f"Warning: Game '{game}' not found in dataset and will be ignored.")
    
    # Remove already rated games from recommendations
    for game in input_games:
        if game in scores:
            scores.drop(game, inplace=True)
    
    # Return top N recommendations
    return scores.nlargest(top_n).index.tolist()

# Input for user_id, is_recommended, and games
print("Available games in the dataset:")
print(", ".join(game_df['title'].sample(20, random_state=42).tolist()))  # Display random sample of 20 games for reference

# Input for user_id
user_id = int(input("\nPlease input your user ID: "))

# Input for games and recommendations
input_games = []
user_recommendations = []

for i in range(5):
    game = input(f"Enter game title {i + 1} (exactly as it appears above): ")
    is_recommended = int(input(f"Enter recommendation for {game} (1 for recommended, 0 for not recommended): "))
    input_games.append(game)
    user_recommendations.append((user_id, game, is_recommended))

# Add the user's input to the recommendation dataset
user_input_df = pd.DataFrame(user_recommendations, columns=['user_id', 'title', 'is_recommended'])
user_input_df = user_input_df.merge(game_df[['app_id', 'title']], on='title', how='left')

# Append the user's input to the existing recommendation dataset
recommendation_df = pd.concat([recommendation_df, user_input_df[['user_id', 'app_id', 'is_recommended']]])

# Recreate the user-game matrix with the updated data
user_game_matrix = recommendation_df.pivot_table(
    index='user_id', columns='title', values='is_recommended', fill_value=0
)

# Compute cosine similarity (game-based collaborative filtering)
game_similarity = cosine_similarity(user_game_matrix.T)
game_similarity_df = pd.DataFrame(game_similarity, index=user_game_matrix.columns, columns=user_game_matrix.columns)

# Show the user's input games
print("\nUser Input Games:")
for game in input_games:
    print(f"- {game}")

# Generate recommendations based on user input
recommended_games = recommend_games(input_games, game_similarity_df)

# Display recommended games
print("\nRecommended Games:")
for game in recommended_games:
    print(game)
