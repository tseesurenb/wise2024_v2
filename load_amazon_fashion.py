#%%
import pandas as pd

average_rating_num = 0

# Define the path to the JSON Lines file
file_path = 'data/amazon/Amazon_Fashion.jsonl'

# Load the data into a DataFrame
df = pd.read_json(file_path, lines=True)

# Select the desired columns
selected_columns = df[['user_id', 'asin', 'rating', 'timestamp']]

# Rename 'asin' to 'item_id'
selected_columns.rename(columns={'asin': 'item_id'}, inplace=True)

# Create unique IDs for users and items
selected_columns['user_id'] = selected_columns['user_id'].astype('category').cat.codes
selected_columns['item_id'] = selected_columns['item_id'].astype('category').cat.codes

# Convert timestamp to a numeric format (e.g., Unix epoch time)
selected_columns['timestamp'] = pd.to_datetime(selected_columns['timestamp'], unit='ms').astype('int64') // 10**9

# Get the number of unique users and items
num_users = selected_columns['user_id'].nunique()
num_items = selected_columns['item_id'].nunique()

print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")

# Calculate the average number of items rated per user
average_items_per_user = selected_columns.groupby('user_id')['item_id'].nunique().mean()

print(f"Average number of items rated per user: {average_items_per_user:.2f}")

# Group by user_id and count unique items rated
user_item_counts = selected_columns.groupby('user_id')['item_id'].nunique()

# Filter users who rated more than 10 items
filtered_users = user_item_counts[user_item_counts > average_rating_num].index
filtered_df = selected_columns[selected_columns['user_id'].isin(filtered_users)]

print(filtered_df)
# %%
average_items_per_user = filtered_df.groupby('user_id')['item_id'].nunique().mean()
print(f"Average number of items rated per user: {average_items_per_user:.2f}")
# %%
# Save the filtered DataFrame to a CSV file
filtered_df.to_csv('data/amazon/amazon_fashion.csv', index=False)

# %%
