# %%
import pandas as pd

# Read the text file into a DataFrame
df = pd.read_csv('data/epinion/rating_with_timestamp.txt', delim_whitespace=True, header=None)

# Assign column names
df.columns = ['userId', 'productId', 'categoryId', 'rating', 'helpfulness', 'timestamp']

# Select only the columns we need
df = df[['userId', 'productId', 'rating', 'timestamp']]

# Rename the columns to match the desired format
df = df.rename(columns={'productId': 'itemId'})

# Display the DataFrame
print(df)

# %%
# Count the number of unique users and items
num_users = df['userId'].nunique()
num_items = df['itemId'].nunique()

print(f"Number of unique users: {num_users}")
print(f"Number of unique items: {num_items}")
# %%
len(df)
# %%
