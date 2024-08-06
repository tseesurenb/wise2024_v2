import pandas as pd
import json

# Define the path to your JSON file
json_file_path = 'data/yelp/yelp_academic_dataset_review.json'
csv_file_path = 'data/yelp/yelp_reviews.csv'

# Initialize an empty list to store the records
records = []

# Read the JSON file line by line and parse the required fields
with open(json_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line)
        filtered_record = {
            'user_id': record['user_id'],
            'item_id': record['business_id'],
            'rating': record['stars'],
            'timestamp': record['date']
        }
        records.append(filtered_record)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(records)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False, sep=',')

print(f'Data has been saved to {csv_file_path}')
