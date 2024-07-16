import torch
import numpy as np
import argparse
import data_loader
import utils
import models
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--hid_size', type=int, default=108)
parser.add_argument('--impute_weight', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

def load_model():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight)
    model.load_state_dict(torch.load('./result/{}_model_less_features_started.pth'.format(args.model)))

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def impute_entire_dataset(model, dataset_iter):
    model.eval()
    imputations_filtered = []
    ground_truths_filtered = []
    special_imputations = []
    special_ground_truths = []
    task_ids_filtered = []


    with torch.no_grad():
        for idx, data in enumerate(dataset_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, None)

            imputation = ret['imputations'].data.cpu().numpy()
            evals = ret['evals'].data.cpu().numpy()

            eval_masks = ret['eval_masks'].data.cpu().numpy()
            masks = ret['masks'].data.cpu().numpy()
            task_ids = ret['task_ids'].data.cpu().numpy()

            # Iterate through each sample in the batch
            for i in range(imputation.shape[0]):
                # Iterate through each time step in the sample
                for t in range(imputation.shape[1]):
                    # Check if any value was originally missing in this time step
                    if np.any(eval_masks[i, t]):
                        imputations_filtered.append(imputation[i, t])
                        ground_truths_filtered.append(evals[i, t])
                        task_ids_filtered.append(task_ids[i, t])  # Add corresponding task_id

                        # Check for masks = 1 and eval_masks = 0
                    if np.any((masks[i, t] == 0) & (eval_masks[i, t] == 0)):
                        special_imputations.append(imputation[i, t])
                        special_ground_truths.append(evals[i, t])

    # Convert to 2D arrays

    imputations_filtered = np.vstack(imputations_filtered)
    ground_truths_filtered = np.vstack(ground_truths_filtered)
    task_ids_filtered = np.array(task_ids_filtered)

    special_imputations = np.vstack(special_imputations)
    special_ground_truths = np.vstack(special_ground_truths)

    return imputations_filtered, ground_truths_filtered, task_ids_filtered, special_imputations, special_ground_truths

# def impute_entire_dataset(model, dataset_iter):
#     model.eval()
#     imputations = []
#     ground_truths = []
#
#     with torch.no_grad():
#         for idx, data in enumerate(dataset_iter):
#             data = utils.to_var(data)
#             ret = model.run_on_batch(data, None)
#
#             imputation = ret['imputations'].data.cpu().numpy()
#             evals = ret['evals'].data.cpu().numpy()
#
#             imputations.append(imputation)
#             ground_truths.append(evals)
#
#     imputations = np.concatenate(imputations, axis=0)
#     ground_truths = np.concatenate(ground_truths, axis=0)
#
#     return imputations, ground_truths

if __name__ == '__main__':
    model = load_model()

    # Load the combined dataset
    combined_dataset_iter = data_loader.get_combined_dataset(batch_size=args.batch_size)
    imputations, ground_truths, task_ids, special_imputations, special_ground_truths = impute_entire_dataset(model, combined_dataset_iter)

    print(imputations.shape[0])
    print(ground_truths.shape[0])
    print(task_ids.shape[0])
    print(special_imputations.shape[0])
    print(special_ground_truths.shape[0])

    # Save the imputations for the entire dataset
    np.save('./result/{}_entire_data_less_features_started.npy'.format(args.model), imputations)

    # Save the ground truth values as numpy array
    np.save('./result/{}_entire_ground_truth_less_features_started.npy'.format(args.model), ground_truths)

    np.save('./result/{}_entire_task_ids_less_features_started.npy'.format(args.model), task_ids)


    # Save the special imputations and ground truth values separately
    #np.save('./result/{}_special_imputations_less_features.npy'.format(args.model), special_imputations)
    #np.save('./result/{}_special_ground_truths_less_features.npy'.format(args.model), special_ground_truths)
