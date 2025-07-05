"""# Use Case: Applying Association Rule Mining for Personalized Movie Genre Recommendations: A Data-Driven Approach"""

!pip install apyori

import pandas as pd
from apyori import apriori

# Sample data
data = {
    'Action': [1, 1, 1, 0, 0, 1, 1, 0, 1],
    'Comedy': [1, 1, 0, 1, 0, 0, 1, 1, 0],
    'Drama': [1, 0, 1, 1, 1, 0, 0, 0, 1],
    'Horror': [0, 0, 1, 0, 1, 0, 1, 0, 1],
    'Romance': [0, 1, 0, 1, 0, 1, 0, 1, 0],
    'Sci-Fi': [1, 0, 0, 1, 1, 0, 0, 1, 0],
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert to list of transactions
transactions = []
for _, row in df.iterrows():
    transactions.append([genre for genre in row.index if row[genre] == 1])

# Apply Apriori
rules = list(apriori(transactions, min_support=0.3, min_confidence=0.7, min_lift=1.0))

# Print rules
print("Association Rules:\n")
for rule in rules:
    for stat in rule.ordered_statistics:
        base = list(stat.items_base)
        add = list(stat.items_add)
        if base and add:
            print(f"Rule: {', '.join(base)} => {', '.join(add)}")
            print(f"Support: {rule.support:.2f}")
            print(f"Confidence: {stat.confidence:.2f}")
            print(f"Lift: {stat.lift:.2f}")
            print("------")

# Recommendation function
def recommend_movies(user_preferences):
    recommended = set()

    for rule in rules:
        for stat in rule.ordered_statistics:
            antecedent = set(stat.items_base)
            consequent = set(stat.items_add)
            if antecedent.issubset(user_preferences):
                recommended.update(consequent)

    # Exclude already watched
    return list(recommended - set(user_preferences))

# Example: User who watched Action & Comedy
user_preferences = ['Action', 'Comedy']
recommended = recommend_movies(user_preferences)

print(f"\nRecommended genres for user preferences {user_preferences}: {recommended}")