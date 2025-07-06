"""
Sentiment Analysis
This script demonstrates: Sentiment Analysis
"""

# Sample posts from social media

social_media_posts = [

    "I really love this new phone! It's fantastic!",

    "This is the worst experience I've ever had. Totally disappointing.",

    "Not sure if I like the new update. It's kinda confusing.",

    "Yay! Finally got my order! So happy right now! 😊"

]



# Get sentiment predictions for the posts

analysis_results = sentiment_analyzer(social_media_posts)



# Print the results for each post

for post, analysis in zip(social_media_posts, analysis_results):

    print(f"Post: