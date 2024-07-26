
#import streamlit as st
#import pandas as pd
#import numpy as np

#st.title('Simple Streamlit App')

#st.write("Here's a simple example to demonstrate Streamlit's capabilities.")

# Slider widget
#x = st.slider('Select a value for x', 0, 100, 50)
#st.write(f'Selected value: {x}')

# DataFrame display
#data = pd.DataFrame({
#    'Column 1': np.random.randn(10),
#    'Column 2': np.random.randn(10)
#})
#st.write('Here is a random dataframe:')
#st.write(data)

# Line chart
#st.line_chart(data)
#%%
from pathlib import Path
from utils import predict, load_model
import dataPrep as dp

from model import LGCN 

g_model = 'lgcn_b_ar'
g_dataset = 'ml100k'
g_verbose = False

print(f'loading {g_dataset} ...')
# load the dataset
rating_df, num_users, num_items, g_mean_rating = dp.load_data(dataset=g_dataset, verbose=g_verbose)

pre_model_path = "./models/ml100k/"

pre_model_size = Path(pre_model_path + "_model.pt").stat().st_size / (1024 * 1024)

print(f"Pretrained model size: {pre_model_size:0.3f} MB")

device  = "cpu"

num_users = 943
num_items = 1682
g_model = 'lgcn_b_ar'

pre_model = LGCN(num_users=num_users,
                num_items=num_items,
                num_layers=3,
                embedding_dim = 64,
                add_self_loops = True,
                mu = 3.5,
                model=g_model,
                u_stats = None,
                verbose=False)


pre_model = load_model(pre_model, pre_model_path)

#
u_id = 10
i_id = 5

# List to store predictions with item IDs
pred = []



# Loop through all items and get predictions for the specified user
for i in range(num_items - 1):
    rating = pre_model.predict(u_id=u_id, i_id=i-1)
    pred.append((i, rating.item()))  # Store item ID and rating as a tuple

# Sort the predictions by rating in descending order
pred.sort(key=lambda x: x[1], reverse=True)

# Get the top 50 predictions
top_K = pred[:20]

# Print the top 50 item IDs and their corresponding predicted ratings
for item_id, rating in top_K:
    print(f"Item ID: {item_id: 05d}, Predicted Rating: {rating: .2f}")
# %%
rating_df
# %%
filtered_df = rating_df[rating_df['userId'] == u_id]
pred = []

# Loop through the filtered DataFrame and get predictions for the user-item pairs
for index, row in filtered_df.iterrows():
    item_id = row['itemId']
    actual_rating = row['rating']
    predicted_rating = pre_model.predict(u_id=u_id, i_id=item_id).item()
    pred.append((item_id, actual_rating, predicted_rating))
    
# Sort the predictions by predicted rating in descending order
pred.sort(key=lambda x: x[2], reverse=True)

# Print the item IDs, actual ratings, and predicted ratings with formatting
for item_id, actual_rating, predicted_rating in pred:
    print(f"Item ID: {item_id:05d}, Actual Rating: {actual_rating:.4f}, Predicted Rating: {predicted_rating:.4f}")
# %%

import numpy as np

def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(k)

def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))

def ndcg_at_k(actual, predicted, k):
    dcg = 0.0
    for i in range(k):
        if predicted[i] in actual:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0

# %%
# Extract actual and predicted item lists
actual_items = filtered_df['itemId'].tolist()
predicted_items = [x[0] for x in pred]

# Define the value of k
k = 30

# Calculate precision, recall, and NDCG
precision = precision_at_k(actual_items, predicted_items, k)
recall = recall_at_k(actual_items, predicted_items, k)
ndcg = ndcg_at_k(actual_items, predicted_items, k)

print(f"Precision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")
print(f"NDCG@{k}: {ndcg:.4f}")
# %%
