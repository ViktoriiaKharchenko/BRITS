import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load normalization parameters
mean = np.load('mean.npy')
std = np.load('std.npy')

# Load the imputed data
imputed_data_normalized = np.load('./result/brits_data.npy')
print(imputed_data_normalized)
# Denormalize the imputed data
def denormalize_data(normalized_data, mean, std):
    return (normalized_data * std) + mean

imputed_data_denormalized = denormalize_data(imputed_data_normalized, mean, std)

val_series_ground_truth = np.load('val_series_ground_truth.npy')

column_index = 4  # Replace with the actual index of STARTED_DATE_hours in your data

# Ensure we are extracting only one column
imputed_started_date_hours = imputed_data_denormalized[:, :, column_index].flatten()
ground_truth_started_date_hours = val_series_ground_truth[:, :, column_index].flatten()


column_index = -5

# Ensure we are extracting only one column
imputed_duration_minutes_3 = imputed_data_denormalized[:, :, column_index].flatten()
ground_truth_duration_minutes_3 = val_series_ground_truth[:, :, column_index].flatten()

# Calculate accuracy metrics
mae_started_date_hours = mean_absolute_error(ground_truth_started_date_hours, imputed_started_date_hours)
mae_duration_minutes_3 = mean_absolute_error(ground_truth_duration_minutes_3, imputed_duration_minutes_3)
#
print(f"Mean Absolute Error for STARTED_DATE_hours: {mae_started_date_hours}")
print(f"Mean Absolute Error for DURATION_MINUTES_3: {mae_duration_minutes_3}")