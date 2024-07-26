
# %%
import dataPrep as dp
g_dataset = 'amazon'
g_verbose = False

print(f'loading {g_dataset} ...')
# load the dataset
rating_df, user_df, item_df, rating_stat = dp.load_data2(dataset=g_dataset, verbose=g_verbose)

print(rating_df.head())

rating_df
# %%
import numpy as np

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
    return 0.

def ndcg_at_k(relevance_scores, k):
    dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(relevance_scores, k) / dcg_max

# Example relevance scores
relevance_scores = [3, 2, 3, 0, 1, 2]
k = 6

ndcg = ndcg_at_k(relevance_scores, k)
print(f'NDCG@{k}: {ndcg}')

# %%
import numpy as np

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

# Example usage:
true_ratings = [5, 4, 3, 2, 1]  # Example true ratings (in descending order of relevance)
predicted_ratings = [4, 5, 2, 3, 1]  # Example predicted ratings
k = len(true_ratings)  # Evaluate NDCG up to the number of items

ndcg = ndcg_at_k(true_ratings, predicted_ratings, k)
print(f"NDCG@{k}: {ndcg}")

# %%
