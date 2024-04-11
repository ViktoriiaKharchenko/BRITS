# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import json


data_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/train_with_missing'
ground_truth_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/train_groundtruth'

test_data = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/test_with_missing'
test_truth_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/test_groundtruth'

patient_ids = []

# Function to calculate mean and std, excluding missing values
def calculate_statistics(data_path, patient_ids):
    numeric_cols = None
    all_data = []

    for patient_id in patient_ids:
        data_file = os.path.join(data_path, f'{patient_id}.csv')
        if os.path.isfile(data_file):
            data = pd.read_csv(data_file, na_values="NA")  # Handle "NA" as missing values
            data = data.drop(columns=['CHARTTIME'], errors='ignore')

            if numeric_cols is None:
                # Determine numeric columns from the first file
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            all_data.append(data[numeric_cols])

    all_data_concatenated = pd.concat(all_data)

    # Compute mean and std, ignoring missing values
    mean = all_data_concatenated.mean(skipna=True)
    std = all_data_concatenated.std(ddof=0, skipna=True)  # Use ddof=0 for population std

    return numeric_cols, mean, std

# Simplified extraction of patient IDs
for filename in os.listdir(data_path):
    # Assuming the entire filename (minus the extension) is the patient ID
    patient_id = os.path.splitext(filename)[0]
    # Add to list if it's a digit
    if patient_id.isdigit():
        patient_ids.append(patient_id)

attributes, mean, std = calculate_statistics(data_path, patient_ids)

print(attributes)

print(mean)

print(std)

# Open separate files for training and testing data
fs_train = open('./json/DACMI_train.json', 'w')
fs_test = open('./json/DACMI_test.json', 'w')


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    num_time_points = masks.shape[0]
    # Determine the number of features dynamically from the masks
    num_features = masks.shape[1]

    for h in range(num_time_points):
        if h == 0:
            deltas.append(np.ones(num_features))
        else:
            deltas.append(np.ones(num_features) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def parse_id(id_, file_stream, mean, std):
    data_file = os.path.join(data_path, f'{id_}.csv')
    ground_truth_file = os.path.join(ground_truth_path, '{}.csv'.format(id_))

    if not os.path.exists(ground_truth_file):
        print(f"Ground truth file for {id_} not found.")
        return

    data = pd.read_csv(data_file)
    ground_truth_data = pd.read_csv(ground_truth_file)

    # Rename 'CHARTTIME' column to 'Time'
    data.rename(columns={'CHARTTIME': 'Time'}, inplace=True)
    ground_truth_data.rename(columns={'CHARTTIME': 'Time'}, inplace=True)

    # Extract values directly from the dataframe, assuming all other columns except 'Time' are features
    data = data.drop('Time', axis=1, errors='ignore')
    ground_truth_data = ground_truth_data.drop('Time', axis=1, errors='ignore')

    values = ((data - mean) / std).values
    evals = ((ground_truth_data - mean) / std).values


    masks = ~np.isnan(values)
    eval_masks = ~np.isnan(evals)

    rec = {
        'forward': parse_rec(values, masks, evals, eval_masks, dir_='forward'),
        'backward': parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward'),
        'label': 1  # Adding a label that is always 1
    }

    rec = json.dumps(rec, cls=NumpyEncoder)
    file_stream.write(rec + '\n')


for id_ in patient_ids:
    print(f'Processing patient {id_}')
    try:
        parse_id(id_, fs_train, mean, std)
    except Exception as e:
        print(e)
        continue

fs_train.close()

patient_ids = []

# Simplified extraction of patient IDs
for filename in os.listdir(test_data):
    # Assuming the entire filename (minus the extension) is the patient ID
    patient_id = os.path.splitext(filename)[0]
    # Add to list if it's a digit
    if patient_id.isdigit():
        patient_ids.append(patient_id)

attributes, mean, std = calculate_statistics(test_data, patient_ids)

for id_ in patient_ids:
    print(f'Processing patient {id_}')
    try:
        parse_id(id_, fs_test, mean, std)
    except Exception as e:
        print(e)
        continue

fs_test.close()