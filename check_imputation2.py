import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

# Load normalization parameters
mean = np.load('mean.npy')
std = np.load('std.npy')

# Load the imputed data for the entire dataset
imputed_data_normalized_entire = np.load('./result/brits_test_data_engineering_no_outliers.npy')
ground_truth_entire = np.load('./result/brits_test_ground_truth_engineering_no_outliers.npy')

imputed_all = np.load('./result/brits_all_imputations_engineering_no_outliers.npy')
evals_all = np.load('./result/brits_all_evals_engineering_no_outliers.npy')
task_ids_all = np.load('./result/brits_all_task_ids_engineering_no_outliers.npy')
masks_all = np.load('./result/brits_all_masks_engineering_no_outliers.npy')
evals_masks_all = np.load('./result/brits_all_evals_masks_engineering_no_outliers.npy')

task_ids = np.load('./result/brits_test_task_ids_engineering_no_outliers.npy')

def denormalize_data(normalized_data, mean, std):
    mean = mean
    std = std
    denormalized_data = (normalized_data * std) + mean

    #denormalized_data[:, 0] = np.round(denormalized_data[:, 0])  # Round only the first column

    return denormalized_data

imputed_data_denormalized_entire = denormalize_data(imputed_data_normalized_entire, mean, std)

ground_truth_denormalized_entire = denormalize_data(ground_truth_entire, mean, std)

imputed_all_denormalized = denormalize_data(imputed_all, mean, std)

evals_denormalized = denormalize_data(evals_all, mean, std)

# Convert the data to pandas DataFrame for easier manipulation
imputed_df = pd.DataFrame(imputed_data_denormalized_entire)
ground_truth_df = pd.DataFrame(ground_truth_denormalized_entire)

# Convert the data to pandas DataFrame for easier manipulation
imputed_all_df = pd.DataFrame(imputed_all_denormalized)
evals_all_df = pd.DataFrame(evals_denormalized)

# Add task IDs to the DataFrames
imputed_df['TaskID'] = task_ids
ground_truth_df['TaskID'] = task_ids

# Add task IDs to the DataFrames
imputed_all_df['TaskID'] = task_ids_all
evals_all_df['TaskID'] = task_ids_all

imputed_all_df['Missing'] = masks_all[:, 5]
evals_all_df['Missing'] = masks_all[:, 5]


imputed_all_df['Evals'] = evals_masks_all[:, 5]
evals_all_df['Evals'] = evals_masks_all[:, 5]
# Ensure task IDs are unique
unique_task_ids = np.unique(task_ids)
num_unique_task_ids = len(unique_task_ids)
print(f"Number of unique task_id values: {num_unique_task_ids}")

# Group by task ID and compute the mean for each group
imputed_mean_per_task = imputed_df.groupby(task_ids).mean()
ground_truth_mean_per_task = ground_truth_df.groupby(task_ids).mean()

# task_id_to_exclude = 7228639
# #
# imputed_mean_per_task = imputed_mean_per_task[imputed_mean_per_task.index != task_id_to_exclude]
# ground_truth_mean_per_task = ground_truth_mean_per_task[ground_truth_mean_per_task.index != task_id_to_exclude]


imputed_copy = imputed_mean_per_task.copy()
imputed_copy['ground_truth_duration'] = ground_truth_mean_per_task.iloc[:, 5]
imputed_copy['ground_truth_started'] = ground_truth_mean_per_task.iloc[:, -7]

output_file = 'imputed_data_with_residuals.xlsx'
imputed_copy.to_excel(output_file, index=False)

# Extract the relevant columns (4 and -1) for calculating errors
imputed_duration_minutes_3_per_task = imputed_mean_per_task.iloc[:, 5]
ground_truth_duration_minutes_3_per_task = ground_truth_mean_per_task.iloc[:, 5]
imputed_duration_minutes_3_per_task[imputed_duration_minutes_3_per_task < 0] = 1


imputed_started_date_per_task = imputed_mean_per_task.iloc[:, -7]
ground_truth_started_date_per_task = ground_truth_mean_per_task.iloc[:, -7]

# Create separate DataFrames for the extracted columns
imputed_duration_minutes_3_df = pd.DataFrame({
    'TaskID': imputed_mean_per_task.index,
    'ImputedDurationMinutes3': imputed_duration_minutes_3_per_task,
    'ImputedStartedDate': imputed_started_date_per_task
})

ground_truth_duration_minutes_3_df = pd.DataFrame({
    'TaskID': ground_truth_mean_per_task.index,
    'GroundTruthDurationMinutes3': ground_truth_duration_minutes_3_per_task,
    'GroundTruthStartedDate': ground_truth_started_date_per_task
})

print(ground_truth_duration_minutes_3_df.shape)
print(imputed_duration_minutes_3_df.shape)

unique_task_ids = np.unique(task_ids_all)
num_unique_task_ids = len(unique_task_ids)
print(f"Number of unique task_id values: {num_unique_task_ids}")

# Group by task ID and compute the mean for each group
imputed_all_per_task = imputed_all_df.groupby(task_ids_all).mean()
evals_all_per_task = evals_all_df.groupby(task_ids_all).mean()

print(imputed_all_per_task.shape[0])

# Assuming 'imputed_all_df' is your DataFrame and 'Missing' is the column indicating imputed values
count_imputed = imputed_all_per_task[(imputed_all_per_task['Missing'] == 0) & (imputed_all_per_task['Evals']==0)].shape[0]

print(f"Number of imputed entries where 'Missing' is 1: {count_imputed}")

# Assuming 'evals_all_df' is another DataFrame
count_eval = evals_all_per_task[(evals_all_per_task['Missing'] == 0)& (evals_all_per_task['Evals']==0)].shape[0]

print(f"Number of eval entries where 'Missing' is 1: {count_eval}")

# Filter rows where both 'Missing' and 'Evals' are not 0
filtered_imputed_df = imputed_all_per_task[(imputed_all_per_task['Missing'] != 0) | (imputed_all_per_task['Evals'] != 0)]
filtered_evals_df = evals_all_per_task[(evals_all_per_task['Missing'] != 0) | (evals_all_per_task['Evals'] != 0)]

# Assuming 'started_date' is the third last column in the DataFrame
created_date_column = imputed_all_per_task.columns[-8]
created_date_column_ev = evals_all_per_task.columns[-8]

# Sort the filtered DataFrames by the 'started_date' column
sorted_imputed_df = filtered_imputed_df.sort_values(by=created_date_column)
sorted_evals_df = filtered_evals_df.sort_values(by=created_date_column_ev)


# Assuming 'duration' is the fourth column from the end in the DataFrame
duration_column = sorted_evals_df.columns[5]

# Determine the midpoint to split the data
midpoint = len(sorted_evals_df)//1

# Split original data
first_half_original = sorted_evals_df.iloc[:midpoint]
second_half_original = sorted_evals_df.iloc[midpoint:]

# Split imputed data
first_half_imputed = sorted_imputed_df.iloc[:midpoint]
second_half_imputed = sorted_imputed_df.iloc[midpoint:]

# Plotting the first half
plt.figure(figsize=(10, 6))

#Original data (values not missing)
plt.plot(first_half_original[created_date_column],
         np.where(first_half_original['Missing'] == 1, first_half_original[duration_column], np.nan),
          linestyle='-', color='g', label='Not Missing (Original Duration)')

# Ground truth for missing values (evals)
plt.plot(first_half_original[created_date_column],
         np.where(first_half_original['Missing'] == 0, first_half_original[duration_column], np.nan),
         marker='o', linestyle='-', color='b', label='Missing (Ground Truth)')

# Imputed data
plt.plot(first_half_imputed[created_date_column],
         np.where(first_half_imputed['Missing'] == 0, first_half_imputed[duration_column], np.nan),
         marker='o', linestyle='-', color='r', label='Imputed Duration')

plt.xlabel('Created Date')
plt.ylabel('Duration')
plt.title('Duration over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for the 'duration' column
Q1 = sorted_evals_df[duration_column].quantile(0.25)
Q3 = sorted_evals_df[duration_column].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for identifying outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers from sorted_evals_df
filtered_evals_df = sorted_evals_df[(sorted_evals_df[duration_column] >= lower_bound) &
                                    (sorted_evals_df[duration_column] <= upper_bound)]

# Also filter the same rows in sorted_imputed_df based on the index of filtered_evals_df
filtered_imputed_df = sorted_imputed_df.loc[filtered_evals_df.index]

# Determine the midpoint to split the data
midpoint = len(filtered_evals_df) // 1

# Split filtered data into two halves
first_half_original = filtered_evals_df.iloc[:midpoint]
second_half_original = filtered_evals_df.iloc[midpoint:]
first_half_imputed = filtered_imputed_df.iloc[:midpoint]
second_half_imputed = filtered_imputed_df.iloc[midpoint:]

# Plotting the first half
plt.figure(figsize=(10, 6))

# Original data (values not missing)
plt.plot(first_half_original[created_date_column],
         np.where(first_half_original['Missing'] == 1, first_half_original[duration_column], np.nan),
          linestyle='-', color='g', label='Not Missing (Original Duration)')

# Ground truth for missing values (evals)
plt.plot(first_half_original[created_date_column],
         np.where(first_half_original['Missing'] == 0, first_half_original[duration_column], np.nan),
         marker='o', linestyle='-', color='b', label='Missing (Ground Truth)')

# Imputed data
plt.plot(first_half_imputed[created_date_column],
         np.where(first_half_imputed['Missing'] == 0, first_half_imputed[duration_column], np.nan),
         marker='o', linestyle='-', color='r', label='Imputed Duration')

plt.xlabel('Created Date')
plt.ylabel('Duration')
plt.title('Duration over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Step 1: Calculate Q1, Q3, and IQR
Q1 = np.percentile(ground_truth_duration_minutes_3_per_task, 25)
Q3 = np.percentile(ground_truth_duration_minutes_3_per_task, 75)
IQR = Q3 - Q1

# Step 2: Calculate the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Identify and count outliers
outliers = (ground_truth_duration_minutes_3_per_task < lower_bound) | (ground_truth_duration_minutes_3_per_task > upper_bound)
num_outliers = np.sum(outliers)

non_outlier_indices = ~outliers
ground_truth_no_outliers = ground_truth_duration_minutes_3_per_task[non_outlier_indices]
imputed_no_outliers = imputed_duration_minutes_3_per_task[non_outlier_indices]

# Print the lower bound, upper bound, and number of outliers
print(f"Lower Bound: {lower_bound}")
# Step 4: Remove outliers
print(f"Upper Bound: {upper_bound}")
print(f"Number of Outliers: {num_outliers}")
print(f"Number of Remaining Imputations: {len(ground_truth_no_outliers)}")

# Boxplot of the residuals without outliers
plt.figure(figsize=(10, 6))
plt.boxplot(imputed_duration_minutes_3_per_task, vert=False)
plt.title('Boxplot of DURATION_MINUTES')
plt.xlabel('Residuals')
plt.show()

residuals_duration_minutes_3_per_task = ground_truth_duration_minutes_3_per_task - imputed_duration_minutes_3_per_task
residuals_started_date_per_task = ground_truth_started_date_per_task - imputed_started_date_per_task


# Calculate accuracy metrics for entire dataset
mae_started_date_hours_entire = mean_absolute_error(ground_truth_started_date_per_task, imputed_started_date_per_task)
mae_duration_minutes_3_entire = mean_absolute_error(ground_truth_duration_minutes_3_per_task, imputed_duration_minutes_3_per_task)

print(f"\nEntire Dataset Metrics:")
print(f"  Mean Absolute Error for STARTED_DATE_hours: {mae_started_date_hours_entire}")
print(f"  Mean Absolute Error for DURATION_MINUTES_3: {mae_duration_minutes_3_entire}")

# Calculate additional metrics for entire dataset
mse_started_date_hours_entire = mean_squared_error(ground_truth_started_date_per_task, imputed_started_date_per_task)
rmse_started_date_hours_entire = np.sqrt(mse_started_date_hours_entire)
medae_started_date_hours_entire = median_absolute_error(ground_truth_started_date_per_task, imputed_started_date_per_task)
r2_started_date_hours_entire = r2_score(ground_truth_started_date_per_task, imputed_started_date_per_task)

mse_duration_minutes_3_entire = mean_squared_error(ground_truth_duration_minutes_3_per_task, imputed_duration_minutes_3_per_task)
rmse_duration_minutes_3_entire = np.sqrt(mse_duration_minutes_3_entire)
medae_duration_minutes_3_entire = median_absolute_error(ground_truth_duration_minutes_3_per_task, imputed_duration_minutes_3_per_task)
r2_duration_minutes_3_entire = r2_score(ground_truth_duration_minutes_3_per_task, imputed_duration_minutes_3_per_task)

# Calculate percentiles for residuals
percentiles_started_date_hours = np.percentile(np.abs(residuals_started_date_per_task), [25, 50, 75, 90, 97, 99])
percentiles_duration_minutes_3 = np.percentile(np.abs(residuals_duration_minutes_3_per_task), [25, 50, 75, 90, 97, 99])

variance_started_date_hours = np.var(np.abs(residuals_started_date_per_task))
variance_duration_minutes_3 = np.var(np.abs(residuals_duration_minutes_3_per_task))

# Calculate Standard Deviation of Residuals
std_dev_started_date_hours = np.std(np.abs(residuals_started_date_per_task))
std_dev_duration_minutes_3 = np.std(np.abs(residuals_duration_minutes_3_per_task))

# Calculate Mean Relative Error (MRE)
mre_started_date_hours_entire = np.mean(np.abs((ground_truth_started_date_per_task - imputed_started_date_per_task) / ground_truth_started_date_per_task))
mre_duration_minutes_3_entire = np.mean(np.abs((ground_truth_duration_minutes_3_per_task - imputed_duration_minutes_3_per_task) / ground_truth_duration_minutes_3_per_task))


print(f"  Metrics for STARTED_DATE_hours (Entire Dataset):")
print(f"    Mean Squared Error: {mse_started_date_hours_entire}")
print(f"    Root Mean Squared Error: {rmse_started_date_hours_entire}")
print(f"    Median Absolute Error: {medae_started_date_hours_entire}")
print(f"    Variance of Residuals: {variance_started_date_hours}")
print(f"    Standard Deviation of Residuals: {std_dev_started_date_hours}")
print(f"    Mean Relative Error: {mre_started_date_hours_entire}")
print(f"    R² Score: {r2_started_date_hours_entire}")
print(f"    Percentiles (25th, 50th, 75th, 90th): {percentiles_started_date_hours}")

print(f"  Metrics for DURATION_MINUTES_3 (Entire Dataset):")
print(f"    Mean Squared Error: {mse_duration_minutes_3_entire}")
print(f"    Root Mean Squared Error: {rmse_duration_minutes_3_entire}")
print(f"    Median Absolute Error: {medae_duration_minutes_3_entire}")
print(f"    Variance of Residuals: {variance_duration_minutes_3}")
print(f"    Standard Deviation of Residuals: {std_dev_duration_minutes_3}")
print(f"    Mean Relative Error: {mre_duration_minutes_3_entire}")
print(f"    R² Score: {r2_duration_minutes_3_entire}")
print(f"    Percentiles (25th, 50th, 75th, 90th): {percentiles_duration_minutes_3}")


# Assuming residuals have been calculated and are in the combined_df DataFrame
residuals_duration_minutes_3 = residuals_duration_minutes_3_per_task

# Create a boxplot for residuals of DurationMinutes3
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=residuals_duration_minutes_3)
plt.title('Boxplot of Residuals for Duration Minutes 3')
plt.xlabel('Duration Minutes 3 Residuals')


plt.tight_layout()
plt.show()

q1_duration = residuals_duration_minutes_3.quantile(0.25)
q3_duration = residuals_duration_minutes_3.quantile(0.75)
iqr_duration = q3_duration - q1_duration

# Define the thresholds for outliers (1.5 times the IQR)
threshold_low_duration = q1_duration - 1.5 * iqr_duration
threshold_high_duration = q3_duration + 1.5 * iqr_duration

print(threshold_low_duration, threshold_high_duration)

# Filter outliers in DurationMinutes3 and StartedDate
outliers_duration = imputed_copy[(residuals_duration_minutes_3 < threshold_low_duration) |
                                          (residuals_duration_minutes_3 > threshold_high_duration)]


# Save the outlier data to an Excel file
outliers_file = 'imputed_mean_per_task_outliers.xlsx'
outliers_duration.to_excel(outliers_file, index=True)

# Plot histograms of the residuals
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(residuals_started_date_per_task, bins=50, kde=True)
plt.title('Residuals of STARTED_DATE_hours')
plt.xlabel('Residual')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(residuals_duration_minutes_3_per_task, bins=50, kde=True)
plt.title('Residuals of DURATION_MINUTES_3')
plt.xlabel('Residual')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Scatter plot of residuals
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(ground_truth_started_date_per_task, residuals_started_date_per_task, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Ground Truth (STARTED_DATE_hours)')
plt.xlabel('Ground Truth STARTED_DATE_hours')
plt.ylabel('Residual')

plt.subplot(1, 2, 2)
plt.scatter(ground_truth_duration_minutes_3_per_task, residuals_duration_minutes_3_per_task, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Ground Truth (DURATION_MINUTES_3)')
plt.xlabel('Ground Truth DURATION_MINUTES_3')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()

# Plot of Ground Truth vs Imputed Values
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(ground_truth_started_date_per_task, imputed_started_date_per_task, alpha=0.5)
plt.plot([ground_truth_started_date_per_task.min(), ground_truth_started_date_per_task.max()],
         [ground_truth_started_date_per_task.min(), ground_truth_started_date_per_task.max()], 'r--')
plt.title('Ground Truth vs Imputed Values (STARTED_DATE_hours)')
plt.xlabel('Ground Truth STARTED_DATE_hours')
plt.ylabel('Imputed STARTED_DATE_hours')

plt.subplot(1, 2, 2)
plt.scatter(ground_truth_duration_minutes_3_per_task, imputed_duration_minutes_3_per_task, alpha=0.5)
plt.plot([ground_truth_duration_minutes_3_per_task.min(), ground_truth_duration_minutes_3_per_task.max()],
         [ground_truth_duration_minutes_3_per_task.min(), ground_truth_duration_minutes_3_per_task.max()], 'r--')
plt.title('Ground Truth vs Imputed Values (DURATION_MINUTES_3)')
plt.xlabel('Ground Truth DURATION_MINUTES_3')
plt.ylabel('Imputed DURATION_MINUTES_3')

plt.tight_layout()
plt.show()


# Step 5: Plot the data without outliers
plt.scatter(ground_truth_no_outliers, imputed_no_outliers, alpha=0.5)
plt.plot([ground_truth_no_outliers.min(), ground_truth_no_outliers.max()],
         [ground_truth_no_outliers.min(), ground_truth_no_outliers.max()], 'r--')
plt.title('Ground Truth vs Imputed Values (DURATION_MINUTES_3) Without Outliers')
plt.xlabel('Ground Truth DURATION_MINUTES_3')
plt.ylabel('Imputed DURATION_MINUTES_3')
plt.show()


# # Plot the histogram of residuals with KDE
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals_duration_minutes_3_entire, bins=200, kde=True)
# plt.title('Residuals of DURATION_MINUTES_3')
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
#
# # Set x-axis limit to focus on the central part of the distribution
# plt.xlim(-80000, 200000)
#
#
# plt.show()
#
# # Plot the histogram of residuals with KDE
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals_duration_minutes_3_entire, bins=50, kde=True)
# plt.title('Residuals of DURATION_MINUTES_3')
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
#
# # Set x-axis limit to focus on the central part of the distribution
# plt.xlim(-40000, 70000)
#
#
# plt.show()