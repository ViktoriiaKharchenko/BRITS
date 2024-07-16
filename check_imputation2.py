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
imputed_data_normalized_entire = np.load('./result/brits_entire_data_less_features_started.npy')
ground_truth_entire = np.load('./result/brits_entire_ground_truth_less_features_started.npy')

task_ids = np.load('./result/brits_entire_task_ids_less_features_started.npy')

def denormalize_data(normalized_data, mean, std):
    mean = mean.mean(axis=0)
    std = std.mean(axis=0)
    denormalized_data = (normalized_data * std) + mean

    #denormalized_data[:, 0] = np.round(denormalized_data[:, 0])  # Round only the first column

    return denormalized_data

imputed_data_denormalized_entire = denormalize_data(imputed_data_normalized_entire, mean, std)

ground_truth_denormalized_entire = denormalize_data(ground_truth_entire, mean, std)

# Convert the data to pandas DataFrame for easier manipulation
imputed_df = pd.DataFrame(imputed_data_denormalized_entire)
ground_truth_df = pd.DataFrame(ground_truth_denormalized_entire)

# Add task IDs to the DataFrames
imputed_df['TaskID'] = task_ids
ground_truth_df['TaskID'] = task_ids

# Ensure task IDs are unique
unique_task_ids = np.unique(task_ids)
num_unique_task_ids = len(unique_task_ids)
print(f"Number of unique task_id values: {num_unique_task_ids}")

# Group by task ID and compute the mean for each group
imputed_mean_per_task = imputed_df.groupby(task_ids).mean()
ground_truth_mean_per_task = ground_truth_df.groupby(task_ids).mean()

# Extract the relevant columns (4 and -1) for calculating errors
imputed_duration_minutes_3_per_task = imputed_mean_per_task.iloc[:, 4]
ground_truth_duration_minutes_3_per_task = ground_truth_mean_per_task.iloc[:, 4]

imputed_started_date_per_task = imputed_mean_per_task.iloc[:, -2]
ground_truth_started_date_per_task = ground_truth_mean_per_task.iloc[:, -2]

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


print(f"  Metrics for STARTED_DATE_hours (Entire Dataset):")
print(f"    Mean Squared Error: {mse_started_date_hours_entire}")
print(f"    Root Mean Squared Error: {rmse_started_date_hours_entire}")
print(f"    Median Absolute Error: {medae_started_date_hours_entire}")
print(f"    R² Score: {r2_started_date_hours_entire}")
print(f"    Percentiles (25th, 50th, 75th, 90th): {percentiles_started_date_hours}")

print(f"  Metrics for DURATION_MINUTES_3 (Entire Dataset):")
print(f"    Mean Squared Error: {mse_duration_minutes_3_entire}")
print(f"    Root Mean Squared Error: {rmse_duration_minutes_3_entire}")
print(f"    Median Absolute Error: {medae_duration_minutes_3_entire}")
print(f"    R² Score: {r2_duration_minutes_3_entire}")
print(f"    Percentiles (25th, 50th, 75th, 90th): {percentiles_duration_minutes_3}")



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