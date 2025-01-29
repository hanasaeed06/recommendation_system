import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BLUE = '\033[94m'
GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Ratings data for clothes
ratings_data = {
    'User1': [5, 3, 0, 0, 4],
    'User2': [4, 0, 0, 0, 3],
    'User3': [0, 0, 5, 4, 0],
    'User4': [3, 3, 0, 5, 4],
    'User5': [0, 0, 4, 0, 0],
}
df = pd.DataFrame(ratings_data, index=['Shirt1', 'Shirt2', 'Pants1', 'Pants2', 'Jacket1'])

def get_similar_users(user_id, df, k=2):
    similarities = {}
    for user in df.columns:
        if user != user_id:
            similarity = cosine_similarity([df[user_id]], [df[user]])[0][0]
            similarities[user] = similarity
    similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    return similar_users

def recommend_items(user_id, df, similar_users):
    recommendations = {}
    for user, _ in similar_users:
        rated_items = df[user]
        for item in df.index:
            if df.loc[item, user_id] == 0 and rated_items[item] != 0:
                if item not in recommendations:
                    recommendations[item] = rated_items[item]
                else:
                    recommendations[item] += rated_items[item]
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return recommendations

target_user = 'User1'
similar_users = get_similar_users(target_user, df)
print(f"{BLUE}Top similar users for {target_user}:{RESET}")
for user, similarity in similar_users:
    print(f"{GREEN}{user}:{RESET} Similarity = {similarity:.2f}")

recommendations = recommend_items(target_user, df, similar_users)
print(f"\n{BLUE}Recommendations for {target_user}:{RESET}")
for item, score in recommendations:
    print(f"{CYAN}{item}:{RESET} Score = {score}")
