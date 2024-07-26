'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

from surprise import Reader
from surprise import Dataset
from sklearn import preprocessing
from torch_geometric.data import download_url, extract_zip
import pandas as pd
#import math
import torch
from torch_sparse import SparseTensor
import numpy as np
#import datetime
from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def scatter_emb(rating_df, u_id, model):
    
    #rating_df['date'] = pd.to_datetime(rating_df['timestamp'], unit='s')
    #rating_df.set_index('date')
    
    udf = rating_df[rating_df['userId'] == u_id]
    
    # Create a figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if model == 'lgcn_b_a' or model == 'lgcn_b_ar':
        # Plotting u_abs_decay
        sc1 = ax1.scatter(udf['date'], udf['u_abs_decay'], c=udf['u_abs_decay'], cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('u_abs_decay')
        ax1.set_title('u_abs_decay vs Date for userId 107')
        # Adding colorbars
        cbar1 = fig.colorbar(sc1, ax=ax1)
        cbar1.set_label('u_abs_decay')
   
    if model == 'lgcn_b_r' or model == 'lgcn_b_ar':
        # Plotting u_rel_decay
        sc2 = ax2.scatter(udf['date'], udf['u_rel_decay'], c=udf['u_rel_decay'], cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('u_rel_decay')
        ax2.set_title('u_rel_decay vs Date for userId 107')
        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label('u_rel_decay')
        
    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()
    
# load dataset        
def load_data(dataset = "ml100k", verbose = False):
    
    user_df = None
    item_df = None
    
    if dataset == "ml-1m": #ml1m dataset
        _reader = Reader(line_format="user item rating timestamp", sep="::") 
        _data = Dataset.load_from_file("data/ml-1m/ratings.dat", reader=_reader)
    elif dataset == "ml-10m": #ml10m dataste
        _reader = Reader(line_format="user item rating timestamp", sep="::")
        _data = Dataset.load_from_file("data/ml-10m/ratings.dat", reader=_reader)
    elif dataset == "ml-100k": #ml100k dataset
        _reader = Reader(line_format="user item rating timestamp", sep="\t")
        _data = Dataset.load_from_file("data/ml-100k/u.data", reader=_reader)
    elif dataset == "dummy": #dummy dataset
        _reader = Reader(line_format="user item rating timestamp", sep="\t") 
        _data = Dataset.load_from_file("data/dummy/test.data", reader=_reader)
    elif dataset == "amazon": #dummy dataset
        _reader = Reader(line_format="user item rating timestamp", sep="\t") 
        _data = Dataset.load_from_file("data/amazon/df_modcloth.csv", reader=_reader)
        #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
    rating_df = pd.DataFrame(_data.raw_ratings, columns=['userId', 'itemId', 'rating', 'timestamp'])    
    _lbl_user = preprocessing.LabelEncoder()
    _lbl_movie = preprocessing.LabelEncoder()

    rating_df.userId = _lbl_user.fit_transform(rating_df.userId.values)
    rating_df.itemId = _lbl_movie.fit_transform(rating_df.itemId.values)
        
    num_users = len(rating_df['userId'].unique())
    num_items = len(rating_df['itemId'].unique())
    mean_rating = rating_df['rating'].mean()
    
    rating_stat = {'num_users': num_users, 'num_items': num_items, 'mean_rating': mean_rating}
    
    if verbose:
        print(f"Using {dataset} dataset.")
        print("The max ID of Users:", rating_df.userId.max(), ", The max ID of Items:", rating_df.itemId.max())   
        print(f"#Users: {num_users}, #Items: {num_items}")
        
    return rating_df, user_df, item_df, rating_stat

def load_data2(dataset = "ml100k", verbose = False):
    
    user_df = None
    item_df = None
    ratings_df = None
    rating_stat = None
    
    if dataset == 'ml-latest-small':
        url = f'https://files.grouplens.org/datasets/movielens/{dataset}.zip'
        extract_zip(download_url(url, '../data'), '../data')

        movies_path = f'../data/{dataset}/movies.csv'
        ratings_path = f'../data/{dataset}/ratings.csv'
        
        print(movies_path)
        print(ratings_path)

        # Load the entire ratings dataframe into memory:
        ratings_df = pd.read_csv(ratings_path)[["userId", "movieId", "rating", "timestamp"]]
        
        ratings_df = ratings_df.rename(columns={'movieId': 'itemId'})

        # Load the entire movie dataframe into memory:
        item_df = pd.read_csv(movies_path)
        item_df = item_df.rename(columns={'movieId': 'itemId'})
        item_df = item_df.set_index('itemId')
        
    elif dataset == 'ml-100k':
        url = f'https://files.grouplens.org/datasets/movielens/{dataset}.zip'
        extract_zip(download_url(url, '../data'), '../data')

        # Paths for ML-100k data files
        ratings_path = f'../data/{dataset}/u.data'
        movies_path = f'../data/{dataset}/u.item'
        users_path = f'../data/{dataset}/u.user'
        
        if verbose:
            print(movies_path)
            print(ratings_path)
            print(users_path)

        # Load the entire ratings dataframe into memory
        ratings_df = pd.read_csv(ratings_path, sep='\t', names=["userId", "movieId", "rating", "timestamp"])
        ratings_df = ratings_df[["userId", "movieId", "rating", "timestamp"]]
        ratings_df = ratings_df.rename(columns={'movieId': 'itemId'})
        
        # Load the entire movie dataframe into memory
        genre_columns = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        item_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', names=["itemId", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_columns)
        
        # Create the genres column by concatenating genre names where the value is 1
        item_df['genres'] = item_df[genre_columns].apply(lambda row: '|'.join([genre for genre, val in row.items() if val == 1]), axis=1)
    
        # Keep only the necessary columns
        item_df = item_df[["itemId", "title", "genres"]]
        item_df = item_df.set_index('itemId')
        
        # Load the entire user dataframe into memory 1|24|M|technician|85711
        user_df = pd.read_csv(users_path, sep='|', encoding='latin-1', names=["userId", "age_group", "sex", "occupation", "zip_code"], engine='python')
        user_df = user_df.set_index('userId')
    
    elif dataset == 'ml-1m':
        url = f'https://files.grouplens.org/datasets/movielens/{dataset}.zip'
        extract_zip(download_url(url, '../data'), '../data')

        # Paths for ML-1M data files
        ratings_path = f'../data/{dataset}/ratings.dat'
        movies_path = f'../data/{dataset}/movies.dat'
        users_path = f'../data/{dataset}/users.dat'
        
        if verbose:
            print(movies_path)
            print(ratings_path)
            print(users_path)

        # Load the entire ratings dataframe into memory
        ratings_df = pd.read_csv(ratings_path, sep='::', names=["userId", "movieId", "rating", "timestamp"], engine='python', encoding='latin-1')
        ratings_df = ratings_df[["userId", "movieId", "rating", "timestamp"]]
        ratings_df = ratings_df.rename(columns={'movieId': 'itemId'})
        
        # Load the entire movie dataframe into memory
        item_df = pd.read_csv(movies_path, sep='::', names=["itemId", "title", "genres"], engine='python', encoding='latin-1')
        item_df = item_df.set_index('itemId')
        
        # Load the entire user dataframe into memory UserID::Gender::Age::Occupation::Zip-code -> 1::F::1::10::48067 
        user_df = pd.read_csv(users_path, sep='::', encoding='latin-1', names=["userId", "sex", "age_group", "occupation", "zip_code"], engine='python')
        user_df = user_df.set_index('userId')
        
    elif dataset == 'amazon':
        # Paths for ML-1M data files
        ratings_path = f'data/{dataset}/df_modcloth.csv'
         #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
        if verbose:
            print(ratings_path)

        # Load the entire ratings dataframe into memory
        
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns
        df_selected = df[['item_id', 'user_id', 'rating', 'timestamp']]

        # Rename the columns
        df_selected.columns = ['itemId', 'userId', 'rating', 'timestamp']
        
        # Convert the 'timestamp' column to datetime objects with error handling
        df_selected.loc[:, 'timestamp'] = pd.to_datetime(df_selected['timestamp'], errors='coerce')
        
        # Option 2: Drop rows with NA timestamps
        #df_selected = df_selected.dropna(subset=['timestamp'])
        
        specific_date = pd.to_datetime('2023-01-01')
        df_selected['timestamp'].fillna(specific_date, inplace=True)

        # Fill the missing values with the average date
        #df_selected['timestamp'].fillna(average_date, inplace=True)

        # Convert the datetime objects to Unix timestamps (handling NaT values if any)
        df_selected.loc[:, 'timestamp'] = df_selected['timestamp'].apply(lambda x: x.timestamp() if pd.notnull(x) else None).astype('Int64')

        # Filter users with at least a minimum number of interactions
        min_interactions = 0
        user_interaction_counts = df_selected['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interactions].index

        df_filtered = df_selected[df_selected['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
    
    elif dataset == 'amazon_fashion':
        # Paths for ML-1M data files
        ratings_path = f'data/amazon/amazon_fashion.csv'
         #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
        if verbose:
            print(ratings_path)

        # Load the entire ratings dataframe into memory
        
        df = pd.read_csv(ratings_path, header=0)

        # Select the relevant columns
        df_selected = df[['user_id', 'item_id', 'rating', 'timestamp']]

        # Rename the columns
        df_selected.columns = ['userId', 'itemId', 'rating', 'timestamp']
                
        # Option 2: Drop rows with NA timestamps
        df_selected = df_selected.dropna(subset=['timestamp'])

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_selected.copy()
        
    elif dataset == 'epinion':
        # Paths for ML-1M data files
        ratings_path = f'data/epinion/rating_with_timestamp.txt'
         #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
        if verbose:
            print(ratings_path)

        # Load the entire ratings dataframe into memory
        
        # Read the text file into a DataFrame
        df = pd.read_csv('data/epinion/rating_with_timestamp.txt', delim_whitespace=True, header=None)

        # Assign column names
        df.columns = ['userId', 'productId', 'categoryId', 'rating', 'helpfulness', 'timestamp']

        # Select only the columns we need
        df = df[['userId', 'productId', 'rating', 'timestamp']]

        # Rename the columns to match the desired format
        df = df.rename(columns={'productId': 'itemId'})
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df.copy()
    
    elif dataset == 'douban_book':
        # Paths for ML-1M data files
        ratings_path = f'data/douban/bookreviews_cleaned.txt'
         #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
        if verbose:
            print(ratings_path)

        # Load the entire ratings dataframe into memory
        
        # Read the text file into a DataFrame
        #df = pd.read_csv(ratings_path, delim_whitespace=True)
        
        # Read the text file into a DataFrame
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        # 'book_id' becomes 'itemId', 'user_id' becomes 'userId', 'rating' remains 'rating', 'time' becomes 'timestamp'
        df = df[['user_id', 'book_id', 'rating', 'time']]
        df = df.rename(columns={'user_id': 'userId', 'book_id': 'itemId', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Filter users with at least a minimum number of interactions
        min_interactions = 20
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
        
    elif dataset == 'douban_music':
        # Paths for ML-1M data files
        ratings_path = f'data/douban/musicreviews_cleaned.txt'
         #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
        if verbose:
            print(ratings_path)

        # Load the entire ratings dataframe into memory
        
        # Read the text file into a DataFrame
        #df = pd.read_csv(ratings_path, delim_whitespace=True)
        
        # Read the text file into a DataFrame
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        # 'book_id' becomes 'itemId', 'user_id' becomes 'userId', 'rating' remains 'rating', 'time' becomes 'timestamp'
        # "user_id"	"music_id"	"rating"	"labels"	"comment"	"useful_num"	"time"	"ID"
        df = df[['user_id', 'music_id', 'rating', 'time']]
        df = df.rename(columns={'user_id': 'userId', 'music_id': 'itemId', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
         # Filter users with at least a minimum number of interactions
        min_interactions = 20
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
        
    elif dataset == 'douban_movie':
        # Paths for ML-1M data files
        ratings_path = f'data/douban/moviereviews_cleaned.txt'
         #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
        if verbose:
            print(ratings_path)

        # Load the entire ratings dataframe into memory
        
        # Read the text file into a DataFrame
        #df = pd.read_csv(ratings_path, delim_whitespace=True)
        
        # Read the text file into a DataFrame
        df = pd.read_csv(ratings_path, sep='\t')

        # Select and rename the columns to match the desired format
        # 'book_id' becomes 'itemId', 'user_id' becomes 'userId', 'rating' remains 'rating', 'time' becomes 'timestamp'
        # "user_id"	"music_id"	"rating"	"labels"	"comment"	"useful_num"	"time"	"ID"
        df = df[['user_id', 'movie_id', 'rating', 'time']]
        df = df.rename(columns={'user_id': 'userId', 'movie_id': 'itemId', 'time': 'timestamp'})
        
        # Convert the timestamp column to Unix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Filter users with at least a minimum number of interactions
        min_interactions = 20
        user_interaction_counts = df['userId'].value_counts()
        filtered_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_filtered = df[df['userId'].isin(filtered_users)]

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        ratings_df = df_filtered.copy()
        
    _lbl_user = preprocessing.LabelEncoder()
    _lbl_movie = preprocessing.LabelEncoder()

    if ratings_df is not None:
        ratings_df.userId = _lbl_user.fit_transform(ratings_df.userId.values)
        ratings_df.itemId = _lbl_movie.fit_transform(ratings_df.itemId.values)
            
        num_users = len(ratings_df['userId'].unique())
        num_items = len(ratings_df['itemId'].unique())
        mean_rating = ratings_df['rating'].mean()
        
        rating_stat = {'num_users': num_users, 'num_items': num_items, 'mean_rating': mean_rating}
    
    if verbose:
        print(rating_stat)
    
    return ratings_df, user_df, item_df, rating_stat

# add time distance scaled with beta
def add_u_abs_decay(rating_df, beta = 0.05, method = 'log', verbose = False):
    
    _beta = beta
    _base = 0.000000001
    _win_unit = 24*3600
    
    rating_df["timestamp"] = rating_df["timestamp"].astype("int64")
    _start = rating_df['timestamp'].min()
    _end = rating_df['timestamp'].max()
    
    _max_distance = _end - _start 
    
    if verbose:
        print(f'Time time distance between max and min:{_max_distance}')
        print(f'The a_beta in user absolute drift:{_beta}')
    
    if method == 'linear':
        rating_df['u_abs_decay'] = _base + ((rating_df['timestamp'] - _start) / _win_unit)
    if method == 'log':
        rating_df['u_abs_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _win_unit), _beta)
    if method == 'log_old':
        rating_df['u_abs_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _win_unit), _beta)
    if method == 'recip':
        rating_df['u_abs_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _win_unit), 1/_beta)
    if method == 'exp':
        rating_df['u_abs_decay'] = _base + np.exp(-_beta * (rating_df['timestamp'] - _start) / _win_unit)
    if method == 'sigmoid':
        rating_df['u_abs_decay'] = sigmoid(_base + (rating_df['timestamp'] - _start) / _win_unit)
    
    if verbose:            
        print(f'The absolute decay method is {method} with param: {beta}')
    
    return rating_df

def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    
    return z

# convert timestamp to day, week, month level
def add_u_rel_decay(rating_df, beta = 25, win_size = 1, method = 'exp', verbose = False):
    
    _beta = beta
    _base = 0.000000001
    _win_unit = 24 * 3600 * win_size
    _plot = False
    
    if verbose:
        print(f'The r_beta in user relative drift:{_beta}, win:{_win_unit}')
        
    # Step 1: Convert timestamp to int64
    rating_df["timestamp"] = rating_df["timestamp"].astype("int64")

    # Step 2: Calculate the minimum timestamp for each userId
    min_timestamp_per_user = rating_df.groupby('userId')['timestamp'].min().reset_index()
    min_timestamp_per_user.columns = ['userId', 'min_timestamp']

    # Step 3: Merge the minimum timestamps back into the original DataFrame
    rating_df = pd.merge(rating_df, min_timestamp_per_user, on='userId')

    # Step 4: Calculate the 'u_rel_decay' column
    if method == 'log':
        rating_df['u_rel_decay'] = _base + np.power(((rating_df['timestamp'] - rating_df['min_timestamp']) / _win_unit), _beta)
    elif method == 'exp':
        rating_df['u_rel_decay'] = _base + np.exp(-_beta * (rating_df['timestamp'] - rating_df['min_timestamp']) / _win_unit)
    if method == 'sigmoid':
        rating_df['u_abs_decay'] = sigmoid(_base + (rating_df['timestamp'] - rating_df['min_timestamp']) / _win_unit)
    
    # Step 5: Drop the 'min_timestamp' column if you no longer need it
    rating_df = rating_df.drop('min_timestamp', axis=1)

    # Now, rating_df contains the new 'u_rel_decay' column
    
    if _plot:
        # Get the sorted values of 'u_rel_decay'
        sorted_values = sorted(rating_df['u_rel_decay'])
        # Plot the sorted values
        plt.plot(sorted_values)
        # Add labels and title
        plt.xlabel('Index (after sorting)')
        plt.ylabel('u_rel_decay')
        plt.title('Sorted Plot of u_rel_decay')
        # Show the plot
        plt.show()
    
    if verbose:
        print(f'The relative decay method is {method} with param: {beta}')
    
    return rating_df
    
# get user stats for each user including number of ratings, mean rating and time distance
def get_user_stats(rating_df, verbose = False):
    
    _dist_unit = 24*3600 # a day
    
    # Group by userId and calculate the required statistics
    user_stats = rating_df.groupby('userId').agg(
        num_ratings=pd.NamedAgg(column='rating', aggfunc='count'),
        mean_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        time_distance=pd.NamedAgg(column='timestamp', aggfunc=lambda x: (x.max() - x.min())/_dist_unit)
    ).reset_index()

    # Rename columns for clarity
    user_stats.columns = ['userId', 'num_ratings', 'mean_rating', 'time_distance']

    return user_stats

# get item stats for each item including number of ratings, mean rating and time distance
def get_item_stats(rating_df, verbose = False):
    
    _dist_unit = 24*3600 # a day
    
    # Group by userId and calculate the required statistics
    item_stats = rating_df.groupby('itemId').agg(
        num_ratings=pd.NamedAgg(column='rating', aggfunc='count'),
        mean_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        time_distance=pd.NamedAgg(column='timestamp', aggfunc=lambda x: (x.max() - x.min())/_dist_unit)
    ).reset_index()

    # Rename columns for clarity
    item_stats.columns = ['itemId', 'num_ratings', 'mean_rating', 'time_distance']

    return item_stats

# get edge ids and edge values between users (source) and items (dest)
def get_rmat_values(rating_df, rating_threshold = 0, verbose = False):

    rating_df["timestamp"] = rating_df["timestamp"].astype("int64")
    
    _src = [user_id for user_id in rating_df["userId"]]
    _dst = [item_id for item_id in rating_df["itemId"]]
    _link_vals = rating_df["rating"].values
    
    _ts = rating_df["timestamp"].values
    
    if "u_abs_decay" in rating_df.columns:
        _abs_decay = rating_df["u_abs_decay"].values
        if verbose:
            print('The u_abs_decay exists.')
    else:
        _abs_decay = rating_df["timestamp"].values
        if verbose:
            print('The u_abs_decay does not exists.')
    
    if "u_rel_decay" in rating_df.columns:
        _rel_decay = rating_df["u_rel_decay"].values
        if verbose:
            print('The u_rel_decay exists.')
    else:
        _rel_decay = rating_df["timestamp"].values
        if verbose:
            print('The u_rel_decay does not exists.')
    
    _true_edges = torch.from_numpy(rating_df["rating"].values).view(-1, 1).to(torch.long) >= rating_threshold
    
    rmat_index = [[],[]]
    rmat_values = []
    rmat_ts = []
    rmat_abs_decay = []
    rmat_rel_decay = []
    
    for i in range(_true_edges.shape[0]):
        if _true_edges[i]:
            rmat_index[0].append(_src[i])
            rmat_index[1].append(_dst[i])
            rmat_values.append(_link_vals[i])
            rmat_ts.append(_ts[i])
            rmat_abs_decay.append(_abs_decay[i])
            rmat_rel_decay.append(_rel_decay[i])
        
    rmat_data = {'rmat_index': torch.tensor(rmat_index), 
                 'rmat_values': torch.tensor(rmat_values), 
                 'rmat_ts': torch.tensor(rmat_ts), 
                 'rmat_abs_t_decay': torch.tensor(rmat_abs_decay), 
                 'rmat_rel_t_decay': torch.tensor(rmat_rel_decay)}
    
    return rmat_data



def train_test_split_by_user2(rmat_data, test_size=0.1, seed=0, verbose=False):
    
    r_idx = rmat_data['rmat_index']
    r_vals = rmat_data['rmat_values']
    r_ts = rmat_data['rmat_ts']
    r_abs_t_decay = rmat_data['rmat_abs_t_decay']
    r_rel_t_decay = rmat_data['rmat_rel_t_decay']
    
    num_users = len(np.unique(r_idx[0]))
    train_r_idx, train_r_vals, train_r_ts, train_r_abs_t_decay, train_r_rel_t_decay = [], [], [], [], []
    val_r_idx, val_r_vals, val_r_ts, val_r_abs_t_decay, val_r_rel_t_decay = [], [], [], [], []

    for user in range(num_users):
        # Find interactions for the current user
        user_interactions = np.where(r_idx[0] == user)[0]

        if len(user_interactions) <= 1:
            # If the user has 0 or 1 interaction, add it all to the training set
            train_indices = user_interactions
            val_test_indices = []
        else:
            # Split interactions for the current user
            train_indices, val_test_indices = train_test_split(user_interactions, test_size=test_size, random_state=seed)

        # Split the edge index and values for the current user
        train_r_idx.append(r_idx[:, train_indices])
        train_r_vals.append(r_vals[train_indices])
        train_r_ts.append(r_ts[train_indices])
        train_r_abs_t_decay.append(r_abs_t_decay[train_indices])
        train_r_rel_t_decay.append(r_rel_t_decay[train_indices])
        
        if len(val_test_indices) > 0:
            val_r_idx.append(r_idx[:, val_test_indices])
            val_r_vals.append(r_vals[val_test_indices])
            val_r_ts.append(r_ts[val_test_indices])
            val_r_abs_t_decay.append(r_abs_t_decay[val_test_indices])
            val_r_rel_t_decay.append(r_rel_t_decay[val_test_indices])
    
    # Concatenate the results for all users
    train_r_idx = torch.from_numpy(np.concatenate(train_r_idx, axis=1))
    train_r_vals = torch.from_numpy(np.concatenate(train_r_vals))
    train_r_ts = torch.from_numpy(np.concatenate(train_r_ts))
    train_r_abs_t_decay = torch.from_numpy(np.concatenate(train_r_abs_t_decay))
    train_r_rel_t_decay = torch.from_numpy(np.concatenate(train_r_rel_t_decay))
    
    if len(val_r_idx) > 0:
        val_r_idx = torch.from_numpy(np.concatenate(val_r_idx, axis=1))
        val_r_vals = torch.from_numpy(np.concatenate(val_r_vals))
        val_r_ts = torch.from_numpy(np.concatenate(val_r_ts))
        val_r_abs_t_decay = torch.from_numpy(np.concatenate(val_r_abs_t_decay))
        val_r_rel_t_decay = torch.from_numpy(np.concatenate(val_r_rel_t_decay))
    else:
        val_r_idx = torch.tensor([])
        val_r_vals = torch.tensor([])
        val_r_ts = torch.tensor([])
        val_r_abs_t_decay = torch.tensor([])
        val_r_rel_t_decay = torch.tensor([])

    if verbose:
        print("Train size:", len(train_r_vals), "Val size:", len(val_r_vals), "Train ts size:", len(train_r_ts), "Val ts size:", len(val_r_ts))
        
    train_data = {'rmat_index': train_r_idx, 'rmat_values': train_r_vals.float(), 'rmat_ts': train_r_ts.float(), 'rmat_abs_t_decay': train_r_abs_t_decay.float(), 'rmat_rel_t_decay': train_r_rel_t_decay.float()}
    val_data = {'rmat_index': val_r_idx, 'rmat_values': val_r_vals.float(), 'rmat_ts': val_r_ts.float(), 'rmat_abs_t_decay': val_r_abs_t_decay.float(), 'rmat_rel_t_decay': val_r_rel_t_decay.float()}

    return train_data, val_data


def train_test_split_by_user(e_idx, e_vals, e_ts, e_abs_t_decay, e_rel_t_decay, test_size=0.1, seed=0, verbose=False):
    num_users = len(np.unique(e_idx[0]))
    train_e_idx, train_e_vals, train_e_ts, train_e_abs_t_decay, train_e_rel_t_decay, val_e_idx, val_e_vals, val_e_ts, val_e_abs_t_decay, val_e_rel_t_decay = [], [], [], [], [], [], [], [], [], []

    for user in range(num_users):
        # Find interactions for the current user
        user_interactions = np.where(e_idx[0] == user)[0]

        # Split interactions for the current user
        train_indices, val_test_indices = train_test_split(user_interactions, test_size=test_size, random_state=seed)
    
        # Split the edge index and values for the current user
        train_e_idx.append(e_idx[:, train_indices])
        train_e_vals.append(e_vals[train_indices])
        train_e_ts.append(e_ts[train_indices])
        train_e_abs_t_decay.append(e_abs_t_decay[train_indices])
        train_e_rel_t_decay.append(e_rel_t_decay[train_indices])
        val_e_idx.append(e_idx[:, val_test_indices])
        val_e_vals.append(e_vals[val_test_indices])
        val_e_ts.append(e_ts[val_test_indices])
        val_e_abs_t_decay.append(e_abs_t_decay[val_test_indices])
        val_e_rel_t_decay.append(e_rel_t_decay[val_test_indices])
    
    # Concatenate the results for all users
    train_e_idx =  torch.from_numpy(np.concatenate(train_e_idx, axis=1))
    train_e_vals =  torch.from_numpy(np.concatenate(train_e_vals))
    train_e_ts =  torch.from_numpy(np.concatenate(train_e_ts))
    train_e_abs_t_decay =  torch.from_numpy(np.concatenate(train_e_abs_t_decay))
    train_e_rel_t_decay =  torch.from_numpy(np.concatenate(train_e_rel_t_decay))
    val_e_idx =  torch.from_numpy(np.concatenate(val_e_idx, axis=1))
    val_e_vals =  torch.from_numpy(np.concatenate(val_e_vals))
    val_e_ts =  torch.from_numpy(np.concatenate(val_e_ts))
    val_e_abs_t_decay =  torch.from_numpy(np.concatenate(val_e_abs_t_decay))
    val_e_rel_t_decay =  torch.from_numpy(np.concatenate(val_e_rel_t_decay))

    if verbose:
        print("Train size:", len(train_e_vals), "Val size:", len(val_e_vals), "Train ts size:", len(train_e_ts), "Val ts size:", len(val_e_ts))

    return train_e_idx, train_e_vals, train_e_ts, train_e_abs_t_decay, train_e_rel_t_decay, val_e_idx, val_e_vals, val_e_ts, val_e_abs_t_decay, val_e_rel_t_decay


def rmat_2_adjmat(num_users, num_items, rmat_data):
    #print("num_users:", num_users, "num_items:", num_items)
    r_M = torch.zeros((num_users, num_items))
    t_M = torch.zeros((num_users, num_items))
    abs_d_M = torch.zeros((num_users, num_items))
    rel_d_M = torch.zeros((num_users, num_items))
    #print('memory allocated')

    rmat_index = rmat_data['rmat_index']
    rmat_values = rmat_data['rmat_values']
    rmat_ts = rmat_data['rmat_ts']
    rmat_abs_t_decay = rmat_data['rmat_abs_t_decay']
    rmat_rel_t_decay = rmat_data['rmat_rel_t_decay']
    
    # convert sparse coo format to dense format to get R
    for i in range(len(rmat_index[0])):
        row_idx = rmat_index[0][i]
        col_idx = rmat_index[1][i]
        r_M[row_idx][col_idx] = rmat_values[i]  # r_M is the rating matrix
        t_M[row_idx][col_idx] = rmat_ts[i]  # t_M is the timestamp matrix
        abs_d_M[row_idx][col_idx] = rmat_abs_t_decay[i]  # d_M is the distance matrix
        rel_d_M[row_idx][col_idx] = rmat_rel_t_decay[i]  # d_M is the distance matrix

    # perform the r_mat to adj_mat conversion
    r_M_transpose = torch.transpose(r_M, 0, 1)
    t_M_transpose = torch.transpose(t_M, 0, 1)
    abs_d_M_transpose = torch.transpose(abs_d_M, 0, 1)
    rel_d_M_transpose = torch.transpose(rel_d_M, 0, 1)

    r_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    r_adj_mat[:num_users, num_users:] = r_M.clone()
    r_adj_mat[num_users:,: num_users] = r_M_transpose.clone()
    
    t_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    t_adj_mat[:num_users, num_users:] = t_M.clone()
    t_adj_mat[num_users:,: num_users] = t_M_transpose.clone()
    
    abs_d_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    abs_d_adj_mat[:num_users, num_users:] = abs_d_M.clone()
    abs_d_adj_mat[num_users:,: num_users] = abs_d_M_transpose.clone()

    rel_d_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    rel_d_adj_mat[:num_users, num_users:] = rel_d_M.clone()
    rel_d_adj_mat[num_users:,: num_users] = rel_d_M_transpose.clone()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    r_adj_mat_coo = r_adj_mat.to_sparse_coo()
    edge_index = r_adj_mat_coo.indices()
    edge_values = r_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    t_adj_mat_coo = t_adj_mat.to_sparse_coo()
    t_adj_mat_coo_ts = t_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    abs_d_adj_mat_coo = abs_d_adj_mat.to_sparse_coo()
    abs_d_adj_mat_coo_decay = abs_d_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    rel_d_adj_mat_coo = rel_d_adj_mat.to_sparse_coo()
    rel_d_adj_mat_coo_decay = rel_d_adj_mat_coo.values()
    
    edge_data = {'edge_index': edge_index, 'edge_values': edge_values, 'edge_attr_ts': t_adj_mat_coo_ts, 'edge_attr_abs_decay': abs_d_adj_mat_coo_decay, 'edge_attr_rel_decay': rel_d_adj_mat_coo_decay}
        
    return edge_data

def rmat_2_adjmat_simple(num_users, num_items, rmat_data):
    #print("num_users:", num_users, "num_items:", num_items)
    
    rmat_index = rmat_data['rmat_index']
    rmat_values = rmat_data['rmat_values']
    rmat_ts = rmat_data['rmat_ts']
    rmat_abs_t_decay = rmat_data['rmat_abs_t_decay']
    rmat_rel_t_decay = rmat_data['rmat_rel_t_decay']

    # Initialize lists for edges and their attributes
    edge_index = [[], []]
    edge_values = []
    edge_attr_ts = []
    edge_attr_abs_decay = []
    edge_attr_rel_decay = []

    num_total = num_users + num_items

    # Populate edge lists directly
    for i in range(len(rmat_index[0])):
        user_idx = rmat_index[0][i]
        item_idx = rmat_index[1][i] + num_users  # offset item index

        edge_index[0].append(user_idx)
        edge_index[1].append(item_idx)
        edge_values.append(rmat_values[i])
        edge_attr_ts.append(rmat_ts[i])
        edge_attr_abs_decay.append(rmat_abs_t_decay[i])
        edge_attr_rel_decay.append(rmat_rel_t_decay[i])

        # Add reverse edges
        edge_index[0].append(item_idx)
        edge_index[1].append(user_idx)
        edge_values.append(rmat_values[i])
        edge_attr_ts.append(rmat_ts[i])
        edge_attr_abs_decay.append(rmat_abs_t_decay[i])
        edge_attr_rel_decay.append(rmat_rel_t_decay[i])

    # Convert lists to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_values = torch.tensor(edge_values, dtype=torch.float)
    edge_attr_ts = torch.tensor(edge_attr_ts, dtype=torch.float)
    edge_attr_abs_decay = torch.tensor(edge_attr_abs_decay, dtype=torch.float)
    edge_attr_rel_decay = torch.tensor(edge_attr_rel_decay, dtype=torch.float)

    edge_data = {
        'edge_index': edge_index,
        'edge_values': edge_values,
        'edge_attr_ts': edge_attr_ts,
        'edge_attr_abs_decay': edge_attr_abs_decay,
        'edge_attr_rel_decay': edge_attr_rel_decay
    }

    return edge_data


def rmat_2_adjmat2(num_users, num_items, rmat_data):
    #print("num_users:", num_users, "num_items:", num_items)
    r_M = torch.zeros((num_users, num_items))
    t_M = torch.zeros((num_users, num_items))
    abs_d_M = torch.zeros((num_users, num_items))
    rel_d_M = torch.zeros((num_users, num_items))
    #print('memory allocated')

    rmat_index = rmat_data['rmat_index']
    rmat_values = rmat_data['rmat_values']
    rmat_ts = rmat_data['rmat_ts']
    rmat_abs_t_decay = rmat_data['rmat_abs_t_decay']
    rmat_rel_t_decay = rmat_data['rmat_rel_t_decay']
    
    # convert sparse coo format to dense format to get R
    for i in range(len(rmat_index[0])):
        row_idx = rmat_index[0][i]
        col_idx = rmat_index[1][i]
        r_M[row_idx][col_idx] = rmat_values[i]  # r_M is the rating matrix
        t_M[row_idx][col_idx] = rmat_ts[i]  # t_M is the timestamp matrix
        abs_d_M[row_idx][col_idx] = rmat_abs_t_decay[i]  # d_M is the distance matrix
        rel_d_M[row_idx][col_idx] = rmat_rel_t_decay[i]  # d_M is the distance matrix

    # perform the r_mat to adj_mat conversion
    r_M_transpose = torch.transpose(r_M, 0, 1)
    t_M_transpose = torch.transpose(t_M, 0, 1)
    abs_d_M_transpose = torch.transpose(abs_d_M, 0, 1)
    rel_d_M_transpose = torch.transpose(rel_d_M, 0, 1)

    r_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    r_adj_mat[:num_users, num_users:] = r_M.clone()
    r_adj_mat[num_users:,: num_users] = r_M_transpose.clone()
    
    t_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    t_adj_mat[:num_users, num_users:] = t_M.clone()
    t_adj_mat[num_users:,: num_users] = t_M_transpose.clone()
    
    abs_d_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    abs_d_adj_mat[:num_users, num_users:] = abs_d_M.clone()
    abs_d_adj_mat[num_users:,: num_users] = abs_d_M_transpose.clone()

    rel_d_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    rel_d_adj_mat[:num_users, num_users:] = rel_d_M.clone()
    rel_d_adj_mat[num_users:,: num_users] = rel_d_M_transpose.clone()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    r_adj_mat_coo = r_adj_mat.to_sparse_coo()
    edge_index = r_adj_mat_coo.indices()
    edge_values = r_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    t_adj_mat_coo = t_adj_mat.to_sparse_coo()
    t_adj_mat_coo_ts = t_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    abs_d_adj_mat_coo = abs_d_adj_mat.to_sparse_coo()
    abs_d_adj_mat_coo_decay = abs_d_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    rel_d_adj_mat_coo = rel_d_adj_mat.to_sparse_coo()
    rel_d_adj_mat_coo_decay = rel_d_adj_mat_coo.values()
    
    edge_data = {'edge_index': edge_index, 
                 'edge_values': edge_values, 
                 'edge_attr_ts': t_adj_mat_coo_ts, 
                 'edge_attr_abs_decay': abs_d_adj_mat_coo_decay, 
                 'edge_attr_rel_decay': rel_d_adj_mat_coo_decay}
        
    return edge_data

def adjmat_2_rmat(num_users, num_items, adj_data, verbose = False):
    
    #adj_e_idx = adj_data['r_adj_mat_coo_indices']
    #adj_e_vals = adj_data['r_adj_mat_coo_values']
    #adj_e_ts = adj_data['t_adj_mat_coo_ts']
    #adj_e_abs_decay = adj_data['abs_d_adj_mat_coo_decay']
    #adj_e_rel_decay = adj_data['rel_d_adj_mat_coo_decay']
    
    adj_e_idx = adj_data['edge_index']
    adj_e_vals = adj_data['edge_values']
    adj_e_ts = adj_data['edge_attr_ts']
    adj_e_abs_decay = adj_data['edge_attr_abs_decay']
    adj_e_rel_decay = adj_data['edge_attr_rel_decay']
                               
    r_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                           col=adj_e_idx[1],
                                           value = adj_e_vals,
                                           sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    t_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                              col=adj_e_idx[1], 
                                              value = adj_e_ts,
                                              sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    abs_d_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                              col=adj_e_idx[1], 
                                              value = adj_e_abs_decay,
                                              sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    rel_d_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                              col=adj_e_idx[1], 
                                              value = adj_e_rel_decay,
                                              sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    r_adj_mat = r_sparse_input_edge_index.to_dense()
    t_adj_mat = t_sparse_input_edge_index.to_dense()
    abs_d_adj_mat = abs_d_sparse_input_edge_index.to_dense()
    rel_d_adj_mat = rel_d_sparse_input_edge_index.to_dense()
    
    if verbose:
        print("adj_mat: \n", r_adj_mat)
        
    r_interact_mat = r_adj_mat[:num_users, num_users:]
    t_interact_mat = t_adj_mat[:num_users, num_users:]
    abs_d_interact_mat = abs_d_adj_mat[:num_users, num_users:]
    rel_d_interact_mat = rel_d_adj_mat[:num_users, num_users:]

    r_mat_edge_index = r_interact_mat.to_sparse_coo().indices()
    r_mat_edge_values = r_interact_mat.to_sparse_coo().values()
    r_mat_edge_ts = t_interact_mat.to_sparse_coo().values()
    r_mat_edge_abs_decay = abs_d_interact_mat.to_sparse_coo().values()
    r_mat_edge_rel_decay = rel_d_interact_mat.to_sparse_coo().values()
    
    return r_mat_edge_index, r_mat_edge_values, r_mat_edge_ts, r_mat_edge_abs_decay, r_mat_edge_rel_decay
