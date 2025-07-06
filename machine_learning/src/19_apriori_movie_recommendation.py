# ================================================================
# Chapter 7: Association Rule Mining
# Use Case: Personalized Movie Genre Recommendations
# ================================================================

# If not already installed, uncomment the line below:
# !pip install apyori

import pandas as pd
from apyori import apriori

# --------------------------------------------------
# Step 1: Create Sample Movie Genre Dataset
# --------------------------------------------------

data = {
    'Action':   [1, 1, 1, 0, 0, 1, 1, 0, 1],
    'Comedy':   [1, 1, 0, 1, 0, 0, 1, 1, 0],
    'Drama':    [1, 0, 1, 1, 1, 0, 0, 0, 1],
    'Horror':   [0, 0, 1, 0, 1, 0, 1, 0, 1],
    'Romance':  [0, 1, 0, 1, 0, 1, 0, 1, 0],
    'Sci-Fi':   [1, 0, 0, 1, 1, 0, 0, 1, 0],
}

df = pd.DataFrame(data)

# --------------------------------------------------
# Step 2: Convert DataFrame to List of Transactions
# --------------------------------------------------

transactions = [
    [genre for genre in row.index if row[genre] == 1]
    for _, row in df.iterrows()
]

# --------------------------------------------------
# Step 3: Apply Apriori Algorithm
# --------------------------------------------------

rules = list(apriori(
    transactions,
    min_support=0.3,
    min_confidence=0.7,
    min_lift=1.0
))

# --------------------------------------------------
# Step 4: Display Discovered Association Rules
# --------------------------------------------------

print("🎬 Association Rules:\n")
for rule in rules:
    for stat in rule.ordered_statistics:
        base = list(stat.items_base)
        add = list(stat.items_add)
        if base and add:
            print(f"Rule: {', '.join(base)} => {', '.join(add)}")
            print(f"  Support: {rule.support:.2f}")
            print(f"  Confidence: {stat.confidence:.2f}")
            print(f"  Lift: {stat.lift:.2f}")
            print("------")

# --------------------------------------------------
# Step 5: Define Recommendation Function
# --------------------------------------------------

def recommend_movies(user_preferences):
    """
    Recommend genres based on user preferences using association rules.
    """
    recommended = set()
    for rule in rules:
        for stat in rule.ordered_statistics:
            antecedent = set(stat.items_base)
            consequent = set(stat.items_add)
            if antecedent.issubset(user_preferences):
                recommended.update(consequent)
    return list(recommended - set(user_preferences))  # Exclude already watched

# --------------------------------------------------
# Step 6: Example Recommendation
# --------------------------------------------------

user_preferences = ['Action', 'Comedy']
recommended = recommend_movies(user_preferences)

print(f"\n🎯 Recommended genres for user preferences {user_preferences}: {recommended}")
