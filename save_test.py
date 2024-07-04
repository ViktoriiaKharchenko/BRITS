import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import data_loader
import utils
from main import args
import models

data_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/train_with_missing'
ground_truth_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/train_groundtruth'

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
    rec['lengths'] = np.zeros_like(values)

    return rec


def test(model, savepath):
    test_data = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/test/test_with_missing/1.csv'
    data = pd.read_csv(test_data, na_values="NA")
    model.load_state_dict(torch.load(savepath))

    values = ((data - mean) / std).values

    evals = np.zeros_like(values)

    # Convert to numpy arrays
    masks = ~np.isnan(values)
    eval_masks = np.zeros_like(values, dtype=bool)  # Placeholder

    # Prepare the data structure as expected by the model
    rec = parse_rec(values, masks.astype(int), evals, eval_masks.astype(int), 'forward')
    rec_backward = parse_rec(values[::-1], masks[::-1].astype(int), evals[::-1], eval_masks[::-1].astype(int),
                             'backward')

    # Convert to tensors and add batch dimension
    values_tensor = torch.tensor(rec['values'], dtype=torch.float32).unsqueeze(0)
    masks_tensor = torch.tensor(rec['masks'], dtype=torch.float32).unsqueeze(0)
    deltas_tensor = torch.tensor(rec['deltas'], dtype=torch.float32).unsqueeze(0)
    evals_tensor = torch.tensor(rec['evals'], dtype=torch.float32).unsqueeze(0)
    eval_masks_tensor = torch.tensor(rec['eval_masks'], dtype=torch.float32).unsqueeze(0)
    dummy_labels = torch.tensor([1], dtype=torch.float32).unsqueeze(0)  # Dummy label, e.g., 1, with batch dimension
    is_train = torch.tensor([0], dtype=torch.float32).unsqueeze(0)  # Dummy label, e.g., 1, with batch dimension

    # Assuming your model has a specific method to handle this structured data
    # Make sure the model is in evaluation mode and on the correct device
    model.eval()
    # if torch.cuda.is_available():
    #     model.cuda()
    #     values_tensor = values_tensor.cuda()
    #     masks_tensor = masks_tensor.cuda()
    #     deltas_tensor = deltas_tensor.cuda()
    #     evals_tensor = evals_tensor.cuda()
    #     eval_masks_tensor = eval_masks_tensor.cuda()

    with torch.no_grad():
        # Adjust this line according to how your model's forward method is defined
        # If your model expects a dictionary, you might need to package these tensors into one
        output = model({'values': values_tensor, 'masks': masks_tensor, 'deltas': deltas_tensor,
                        'evals': evals_tensor, 'eval_masks': eval_masks_tensor,
                        'forward': rec, 'backward': rec_backward, 'is_train': is_train,'labels': dummy_labels # Adding a label that is always 1
        })
        imputation = output['imputations'].data.cpu().numpy()
        print(imputation)


model = getattr(models, args.model).Model(13, args.hid_size, args.impute_weight, args.label_weight)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params is {}'.format(total_params))

if torch.cuda.is_available():
    model = model.cuda()

savepath='./result/imputation_model.pt'
test(model,savepath)