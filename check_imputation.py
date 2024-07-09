import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

# Load normalization parameters
mean = np.load('mean.npy')
std = np.load('std.npy')

# Load the imputed data
imputed_data_normalized = np.load('./result/brits_data_2.npy')
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


# Compute the residuals
residuals_started_date_hours = ground_truth_started_date_hours - imputed_started_date_hours
residuals_duration_minutes_3 = ground_truth_duration_minutes_3 - imputed_duration_minutes_3

# Calculate accuracy metrics
mae_started_date_hours = mean_absolute_error(ground_truth_started_date_hours, imputed_started_date_hours)
mae_duration_minutes_3 = mean_absolute_error(ground_truth_duration_minutes_3, imputed_duration_minutes_3)
#
print(f"Mean Absolute Error for STARTED_DATE_hours: {mae_started_date_hours}")
print(f"Mean Absolute Error for DURATION_MINUTES_3: {mae_duration_minutes_3}")

# Calculate additional metrics
mse_started_date_hours = mean_squared_error(ground_truth_started_date_hours, imputed_started_date_hours)
rmse_started_date_hours = np.sqrt(mse_started_date_hours)
medae_started_date_hours = median_absolute_error(ground_truth_started_date_hours, imputed_started_date_hours)
r2_started_date_hours = r2_score(ground_truth_started_date_hours, imputed_started_date_hours)

mse_duration_minutes_3 = mean_squared_error(ground_truth_duration_minutes_3, imputed_duration_minutes_3)
rmse_duration_minutes_3 = np.sqrt(mse_duration_minutes_3)
medae_duration_minutes_3 = median_absolute_error(ground_truth_duration_minutes_3, imputed_duration_minutes_3)
r2_duration_minutes_3 = r2_score(ground_truth_duration_minutes_3, imputed_duration_minutes_3)

print(f"Metrics for STARTED_DATE_hours:")
print(f"  Mean Absolute Error: {mae_started_date_hours}")
print(f"  Mean Squared Error: {mse_started_date_hours}")
print(f"  Root Mean Squared Error: {rmse_started_date_hours}")
print(f"  Median Absolute Error: {medae_started_date_hours}")
print(f"  R² Score: {r2_started_date_hours}")

print(f"\nMetrics for DURATION_MINUTES_3:")
print(f"  Mean Absolute Error: {mae_duration_minutes_3}")
print(f"  Mean Squared Error: {mse_duration_minutes_3}")
print(f"  Root Mean Squared Error: {rmse_duration_minutes_3}")
print(f"  Median Absolute Error: {medae_duration_minutes_3}")
print(f"  R² Score: {r2_duration_minutes_3}")

# Plot histograms of the residuals
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(residuals_started_date_hours, bins=50, kde=True)
plt.title('Residuals of STARTED_DATE_hours')
plt.xlabel('Residual')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(residuals_duration_minutes_3, bins=50, kde=True)
plt.title('Residuals of DURATION_MINUTES_3')
plt.xlabel('Residual')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Scatter plot of residuals
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(ground_truth_started_date_hours, residuals_started_date_hours, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Ground Truth (STARTED_DATE_hours)')
plt.xlabel('Ground Truth STARTED_DATE_hours')
plt.ylabel('Residual')

plt.subplot(1, 2, 2)
plt.scatter(ground_truth_duration_minutes_3, residuals_duration_minutes_3, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Ground Truth (DURATION_MINUTES_3)')
plt.xlabel('Ground Truth DURATION_MINUTES_3')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()
