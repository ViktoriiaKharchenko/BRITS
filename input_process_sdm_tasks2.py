import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


df_ground = pd.read_csv('./sdm_tasks/sdm_tasks_ground_truth.txt')
df_missing = pd.read_csv('./sdm_tasks/sdm_tasks_missing.txt')

# Calculate the percentage of rows with missing values for each file
missing_values_percentage_ground = (df_ground.isnull().any(axis=1).sum() / len(df_ground)) * 100
missing_values_percentage_missing = (df_missing.isnull().any(axis=1).sum() / len(df_missing)) * 100

print(f"Percentage of rows with missing values in ground truth file: {missing_values_percentage_ground:.2f}%")
print(f"Percentage of rows with missing values in missing data file: {missing_values_percentage_missing:.2f}%")

# Create a copy of the missing data to maintain original nulls
combined_data = df_missing.copy()

# Add a stratification column indicating whether DURATION_MINUTES_3 is missing
combined_data['DURATION_MINUTES_3_missing'] = combined_data['DURATION_MINUTES_3'].isnull().astype(int)

# Add ground truth columns for only the columns that can contain missing values
combined_data['STARTED_DATE_hours_ground'] = df_ground['STARTED_DATE_hours']
combined_data['DURATION_MINUTES_3_ground'] = df_ground['DURATION_MINUTES_3']

# Remove rows where the ground truth columns have NaN values
#combined_data.dropna(subset=['STARTED_DATE_hours_ground', 'DURATION_MINUTES_3_ground'], inplace=True)

#combined_data.dropna(subset=['DURATION_MINUTES_3_ground'], inplace=True)

# Remove the `DURATION_MINUTES_3_missing` and `MISSING_FLAG` columns
combined_data = combined_data.drop(columns=['MISSING_FLAG'])

# Sort the combined data based on CREATED_DATE_hours
combined_data_sorted = combined_data.sort_values(by='CREATED_DATE_hours')

print(combined_data_sorted)

# Split the data into training and test sets with stratification
train_data, test_data = train_test_split(combined_data_sorted, test_size=0.2, random_state=42, stratify=combined_data_sorted['DURATION_MINUTES_3_missing'])

# Ensure no overlap by making sure no index in test_data is in train_data
train_indices = set(train_data.index)
test_indices = set(test_data.index)
assert train_indices.isdisjoint(test_indices), "Train and test sets overlap!"

# Calculate the percentage of missing values in DURATION_MINUTES_3 after splitting
train_missing_percentage = train_data['DURATION_MINUTES_3_missing'].mean() * 100
test_missing_percentage = test_data['DURATION_MINUTES_3_missing'].mean() * 100

print(train_data.shape[0])
print(test_data.shape[0])

print(f"Percentage of missing values in DURATION_MINUTES_3 in training set: {train_missing_percentage:.2f}%")
print(f"Percentage of missing values in DURATION_MINUTES_3 in testing set: {test_missing_percentage:.2f}%")

# Remove the `DURATION_MINUTES_3_missing` column
train_data = train_data.drop(columns=['DURATION_MINUTES_3_missing'])
test_data = test_data.drop(columns=['DURATION_MINUTES_3_missing'])

# Sort the combined data based on CREATED_DATE_hours
train_data2 = train_data.sort_values(by='CREATED_DATE_hours')
test_data2 = test_data.sort_values(by='CREATED_DATE_hours')

test_data2.to_csv('test_data_old.csv', index=False)

print(train_data2)

# Create non-random time series samples of 36 consecutive steps for the test set
def sequential_select_time_series(data, sequence_length=15):
    time_series_data = []
    ground_truth_data = []
    for start in range(0, len(data) - sequence_length + 1):
        end = start + sequence_length
        # Select the series data
        time_series = data.iloc[start:end, :29].values

        # Select the ground truth data
        ground_truth = data.iloc[start:end, :29].copy()
        ground_truth.loc[:, 'STARTED_DATE_hours'] = data.iloc[start:end]['STARTED_DATE_hours_ground'].values
        ground_truth.loc[:, 'DURATION_MINUTES_3'] = data.iloc[start:end]['DURATION_MINUTES_3_ground'].values

        time_series_data.append(time_series)
        ground_truth_data.append(ground_truth.values)

    return np.array(time_series_data), np.array(ground_truth_data)

# Form sequences from the split data without overlapping
train_series, train_series_ground_truth = sequential_select_time_series(train_data2)
test_series, test_series_ground_truth = sequential_select_time_series(test_data2)

#np.save('val_series_ground_truth_old.npy', test_series_ground_truth)
#np.save('train_series_ground_truth2.npy', train_series_ground_truth)


print("Train series shape:", train_series.shape)
print("Test series shape:", test_series.shape)
print("Train series ground truth shape:", train_series_ground_truth.shape)
print("Test series ground truth shape:", test_series_ground_truth.shape)



# Normalize features

# Normalize features ignoring NaNs
def normalize_data(train_data, test_data, ground_truth_train, ground_truth_test):
    mean = np.nanmean(train_data, axis=0).mean(axis=0)
    std = np.nanstd(train_data, axis=0).mean(axis=0)

    # Replace zeros in std with 1 to avoid division by zero
    std[std == 0] = 1

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    ground_truth_train = (ground_truth_train - mean) / std
    ground_truth_test = (ground_truth_test - mean) / std
    return train_data, test_data, ground_truth_train, ground_truth_test, mean, std


train_series_normalized, test_series_normalized, train_ground_truth_normalized, test_ground_truth_normalized, mean, std = normalize_data(train_series,
                                test_series, train_series_ground_truth, test_series_ground_truth)

# Save mean and std to files
np.save('mean.npy', mean)
np.save('std.npy', std)


# Function to generate masks and deltas
def generate_masks_and_deltas(values):
    masks = ~np.isnan(values)
    deltas = np.zeros_like(values)
    deltas[0] = 1
    for t in range(1, values.shape[0]):
        deltas[t] = 1 + (1 - masks[t]) * deltas[t - 1]
    return masks, deltas


# Function to process a time series record
def process_record(values, ground_truth):
    masks, deltas = generate_masks_and_deltas(values)
    eval_masks = masks ^ ~np.isnan(ground_truth)

    forwards = pd.DataFrame(values).ffill().fillna(0.0).to_numpy()
    record = {
        'values': np.nan_to_num(values).tolist(),
        'masks': masks.astype('int32').tolist(),
        'evals': np.nan_to_num(ground_truth).tolist(),
        'eval_masks': eval_masks.astype('int32').tolist(),
        'forwards': forwards.tolist(),
        'deltas': deltas.tolist()
    }
    return record


#Process all training records
train_records = []
for i in range(train_series_normalized.shape[0]):
    values = train_series_normalized[i]
    ground_truth = train_ground_truth_normalized[i]
    record = {
        'forward': process_record(values, ground_truth),
        'backward': process_record(values[::-1], ground_truth[::-1]),
        'is_train': 1  # Indicates that this record is from the training set

    }
    train_records.append(record)

# Process all test records
test_records = []
for i in range(test_series_normalized.shape[0]):
    values = test_series_normalized[i]
    ground_truth = test_ground_truth_normalized[i]
    record = {
        'forward': process_record(values, ground_truth),
        'backward': process_record(values[::-1], ground_truth[::-1]),
        'is_train': 0  # Indicates that this record is from the test set

    }
    test_records.append(record)

# Save to JSON files
with open('train.json', 'w') as f:
    json.dump(train_records, f, cls=NpEncoder)

with open('test.json', 'w') as f:
    json.dump(test_records, f, cls=NpEncoder)