#%%
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from sklearn.metrics import ndcg_score

# Load the dataset into Surprise format
#reader = Reader(rating_scale=(1, 5))
#data = Dataset.load_from_df(ml_100k_df[['userId', 'movieId', 'rating']], reader)

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin("ml-1m")
type(data)
#%%
# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print(f"Number of users: {trainset.n_users}")
print(f"Number of items: {trainset.n_items}")
print(f"Number of train ratings: {trainset.n_ratings}")
print(f"Number of test ratings: {len(testset)}")
      

# Train SVD model
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
# Prepare ground truth ratings and predicted ratings
true_ratings = []
predicted_ratings = []

for uid, iid, true_r, est, _ in predictions:
    true_ratings.append(true_r)
    predicted_ratings.append(est)

# Convert lists to numpy arrays
true_ratings = np.array(true_ratings)
predicted_ratings = np.array(predicted_ratings)

# Compute NDCG score
ndcg = ndcg_score(np.expand_dims(true_ratings, axis=0), np.expand_dims(predicted_ratings, axis=0), k=len(testset))
print(f"NDCG Score: {ndcg: .4f}")

def dcg_at_k(r, k):
    """Calculate DCG@k (Discounted Cumulative Gain)"""
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(true_ratings, predicted_ratings, k):
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain)"""
    dcg_max = dcg_at_k(sorted(true_ratings, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(predicted_ratings, k) / dcg_max

#top_k=len(testset)
top_k = 10

ndcg = ndcg_at_k(np.expand_dims(true_ratings, axis=0), np.expand_dims(predicted_ratings, axis=0), k=top_k)
print(f"custom NDCG Score: {ndcg:.4f}")
# %%
