import pandas as pd
import json
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DATA_PATH = './'   # change if your dataset is elsewhere
PROFILE_PATH = 'user_profile.json'
N_CLUSTERS = 5

# ---------------------------------------------------
# Load & preprocess data
# ---------------------------------------------------


print("Loading data...")
movies_df = pd.read_csv(DATA_PATH + 'movies.csv')
ratings_df = pd.read_csv(DATA_PATH + 'ratings.csv')

# Build movie catalog: movieId → {title, genres}
movie_catalog = {}
for _, row in movies_df.iterrows():
    movie_catalog[row['movieId']] = {
        'title': row['title'],
        'genres': row['genres'].split('|')
    }

# Unique list of genres
all_genres = sorted({g for genres in movies_df['genres'].str.split('|') for g in genres})

# ---------------------------------------------------
# Build user feature vectors for clustering
# ---------------------------------------------------

print("⚙ Building user features (low memory)...")

# Make per‑movie genre columns: binary 1/0 if movie has genre
for g in all_genres:
    movies_df[g] = movies_df['genres'].apply(lambda x: int(g in x.split('|')))

# Merge ratings with movies
merged = ratings_df.merge(movies_df[['movieId'] + all_genres], on='movieId')

# Compute per‑user average rating for each genre
user_genre_sum = merged.groupby('userId')[all_genres].sum()
user_genre_count = merged.groupby('userId')[all_genres].count()

user_genre_avg = user_genre_sum / user_genre_count.replace(0, 1)  # avoid div by zero

# Add total_movies_watched and overall_avg_rating
user_stats = merged.groupby('userId').agg(
    total_movies_watched=('movieId','count'),
    overall_avg_rating=('rating','mean')
)

# Combine
user_features_df = user_genre_avg.reset_index().merge(user_stats, on='userId')

print(f" Built features for {len(user_features_df)} users.")


# ---------------------------------------------------
# Train clustering model
# ---------------------------------------------------

print(" Training clustering model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_features_df.drop(columns=['userId']))
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Map userId → cluster
user_clusters_df = pd.DataFrame({
    'userId': user_features_df['userId'],
    'cluster': kmeans.labels_
})

# ---------------------------------------------------
# Recommender functions
# ---------------------------------------------------

def get_current_user_feature_vector(user_profile):
    """Compute feature vector from current user's history"""
    genre_sum = {g:0 for g in all_genres}
    genre_count = {g:0 for g in all_genres}
    total_rating, total_count = 0,0

    for entry in user_profile['history']:
        movie = movie_catalog.get(entry['movieId'])
        if not movie: continue
        rating = entry.get('rating')
        if rating is None: continue
        total_rating += rating
        total_count += 1
        for g in movie['genres']:
            genre_sum[g] += rating
            genre_count[g] += 1

    vec = []
    for g in all_genres:
        vec.append(genre_sum[g]/genre_count[g] if genre_count[g]>0 else 0)
    vec.append(len(user_profile['history']))
    vec.append(total_rating/total_count if total_count>0 else 0)
    return pd.DataFrame([vec])

def get_cluster_based_recommendations(user_profile, top_n=10):
    """Recommend movies popular in the user's cluster"""
    feature_vector = get_current_user_feature_vector(user_profile)
    feature_scaled = scaler.transform(feature_vector)
    cluster_num = kmeans.predict(feature_scaled)[0]
    cluster_users = user_clusters_df[user_clusters_df['cluster']==cluster_num]['userId']
    cluster_ratings = ratings_df[ratings_df['userId'].isin(cluster_users)]

    watched = {e['movieId'] for e in user_profile['history']}
    cluster_ratings = cluster_ratings[~cluster_ratings['movieId'].isin(watched)]

    top_movies = (cluster_ratings.groupby('movieId')['rating']
                  .mean().sort_values(ascending=False).head(top_n).index.values)

    return [dict(movie_catalog[mid], movieId=mid) for mid in top_movies if mid in movie_catalog]

def get_cold_start_movies(feed_size=40):
    """Random unseen movies at start"""
    unseen = [dict(m, movieId=mid) for mid,m in movie_catalog.items()]
    return random.sample(unseen, min(feed_size,len(unseen)))

def generate_feed(user_profile, feed_size=40):
    """Hybrid: 40% recent (last 10, rating weighted), 20% older history, 30% ML, 10% random"""
    if not user_profile['history'] or len(user_profile['history']) < 5:
        return get_cold_start_movies(feed_size)

    watched = {e['movieId'] for e in user_profile['history']}
    recent = user_profile['history'][-10:]
    older = user_profile['history'][:-10]

    related = []

    def rating_multiplier(rating):
        if rating is None:
            return 1
        elif rating >= 5:
            return 2
        elif rating == 4:
            return 1.5
        elif rating <= 2:
            return 0.5
        else:
            return 1

    # recent part
    for entry in recent:
        movie = movie_catalog.get(entry['movieId'])
        if movie:
            mult = rating_multiplier(entry.get('rating'))
            genre_set = set(movie['genres'])
            for mid, m in movie_catalog.items():
                if mid not in watched and genre_set & set(m['genres']):
                    related.extend([dict(m, movieId=mid)] * int(mult * 2))  # duplicate → boost

    # older part
    for entry in older:
        movie = movie_catalog.get(entry['movieId'])
        if movie:
            mult = rating_multiplier(entry.get('rating'))
            genre_set = set(movie['genres'])
            for mid, m in movie_catalog.items():
                if mid not in watched and genre_set & set(m['genres']):
                    related.extend([dict(m, movieId=mid)] * int(mult))  # smaller boost

    # deduplicate
    related = list({m['movieId']: m for m in related}.values())
    random.shuffle(related)

    num_recent = feed_size * 40 // 100
    num_older = feed_size * 20 // 100
    num_ml = feed_size * 30 // 100

    feed = []

    feed.extend(related[:num_recent+num_older])

    # ML part
    if len(user_profile['history']) >= 20:
        feed.extend(get_cluster_based_recommendations(user_profile, top_n=num_ml))

    # Random unseen to fill remaining
    remaining = feed_size - len(feed)
    unseen = [dict(m, movieId=mid) for mid, m in movie_catalog.items() if mid not in watched]
    if remaining > 0 and unseen:
        feed.extend(random.sample(unseen, min(remaining, len(unseen))))

    random.shuffle(feed)
    return feed[:feed_size]


# ---------------------------------------------------
# CLI loop
# ---------------------------------------------------

def cli_session_loop(user_profile):
    print("\n Welcome! Type q to quit.\n")
    while True:
        feed = generate_feed(user_profile)
        print("\n Your Feed:")
        for i,m in enumerate(feed,1):
            print(f"{i}. [{m['movieId']}] {m['title']} ({', '.join(m['genres'])})")
        choice = input("\nPick number (1-40), 0=manual, q=quit: ").strip()
        if choice=='q': break
        elif choice=='0':
            mid = input("Enter movieId: ").strip()
            if mid.isdigit() and int(mid) in movie_catalog:
                movie = movie_catalog[int(mid)]
                print(f"✅ Watched: {movie['title']}")
                rate = input("Rate (1–5) or Enter to skip: ").strip()
                rating = int(rate) if rate.isdigit() and 1<=int(rate)<=5 else None
                user_profile['history'].append({'movieId':int(mid), 'timestamp':pd.Timestamp.now().isoformat(),'rating':rating})
            else:
                print("Not found.")
        elif choice.isdigit() and 1<=int(choice)<=len(feed):
            movie = feed[int(choice)-1]
            print(f"✅ Watched: {movie['title']}")
            rate = input("Rate (1–5) or Enter to skip: ").strip()
            rating = int(rate) if rate.isdigit() and 1<=int(rate)<=5 else None
            user_profile['history'].append({'movieId':int(movie['movieId']), 'timestamp':pd.Timestamp.now().isoformat(),'rating':rating})
        else:
            print("❗ Invalid input.")

# ---------------------------------------------------
# Save/load profile
# ---------------------------------------------------

def save_user_profile(profile, filename=PROFILE_PATH):
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(profile,f,indent=2)
def load_user_profile(filename=PROFILE_PATH):
    try:
        with open(filename,'r',encoding='utf-8') as f:
            print(" User profile loaded.")
            return json.load(f)
    except FileNotFoundError:
        print(" New user profile created.")
        return {'history': []}

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__=="__main__":
    user_profile = load_user_profile()
    cli_session_loop(user_profile)
    save_user_profile(user_profile)
    print(" Bye!")
