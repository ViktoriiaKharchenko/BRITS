import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load normalization parameters
mean = np.load('mean.npy')
std = np.load('std.npy')

# Load the imputed data for the entire dataset
imputed_data_normalized_entire = np.load('./result/brits_special_ground_truths_less_features.npy')
ground_truth_entire = np.load('./result/brits_special_imputations_less_features.npy')

# Function to denormalize data
def denormalize_data(normalized_data, mean, std):
    #mean = mean.mean(axis=0)
    #std = std.mean(axis=0)
    denormalized_data = (normalized_data * std) + mean
    denormalized_data[:, 0] = np.round(denormalized_data[:, 0])  # Round only the first column

    #return np.round(denormalized_data)
    return denormalized_data

# Denormalize the imputed data
imputed_data_denormalized_entire = denormalize_data(imputed_data_normalized_entire, mean, std)

# Convert the denormalized data to DataFrame
imputed_df = pd.DataFrame(imputed_data_denormalized_entire)

# Save the DataFrame to a CSV file
imputed_df.to_csv('./imputed_data/invalid_imputed_data_less_features.csv', index=False)

print("Denormalized imputed data saved to './imputed_data/denormalized_imputed_data_less_features.csv'")

print(imputed_df.shape[0])

# Get the number of unique task_id values
task_ids = imputed_data_denormalized_entire[:, 0]
num_unique_task_ids = len(np.unique(task_ids))
print(f"Number of unique task_id values: {num_unique_task_ids}")

# Perform distribution analysis on the imputed duration (4th column)
duration_column = imputed_data_denormalized_entire[:, 4]
# Calculate the percentage of entries where the value of column 4 is higher than the value of column 3
created_duration = imputed_data_denormalized_entire[:, 3]
condition = duration_column > created_duration
percentage_higher = np.mean(condition) * 100
print(percentage_higher)



# Basic statistics
mean_duration = np.mean(duration_column)
median_duration = np.median(duration_column)
std_duration = np.std(duration_column)

print(f"Distribution Analysis of Imputed Duration:")
print(f"  Mean: {mean_duration}")
print(f"  Median: {median_duration}")
print(f"  Standard Deviation: {std_duration}")

# Plot the distribution of the imputed duration
plt.figure(figsize=(10, 6))
sns.histplot(duration_column, bins=50, kde=True)
plt.title('Distribution of Imputed Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()


# Filter the rows based on the condition
filtered_data = imputed_data_denormalized_entire[condition]

# Save the filtered data to a separate CSV file
filtered_csv_path = './imputed_data/wrong_imputed_data_less_features.csv'

# Convert the denormalized data to DataFrame
imputed_df = pd.DataFrame(filtered_data)

# Save the DataFrame to a CSV file
imputed_df.to_csv(filtered_csv_path, index=False)