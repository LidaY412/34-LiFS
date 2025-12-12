import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from ResNet_3D import ResNet18_3D_7stream_LSTM
from Nii_utils import NiiDataRead


class Dataset_for_external_validation(Dataset):
    """Dataset class for external validation"""
    def __init__(self, data_dir, split_path):
        self.data_dir = data_dir
        with open(split_path, 'r') as f:
            ID_list_original = f.readlines()
        self.ID_list = [n.strip('\n') for n in ID_list_original]
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        
        # Read 7 modal images
        img_1, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '1_img.nii.gz'))
        img_2, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '2_img.nii.gz'))
        img_3, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '3_img.nii.gz'))
        img_4, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '4_img.nii.gz'))
        img_5, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '5_img.nii.gz'))
        img_6, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '6_img.nii.gz'))
        img_7, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '7_img.nii.gz'))

        # Process images
        img = np.concatenate((img_1[..., np.newaxis],
                              img_2[..., np.newaxis],
                              img_3[..., np.newaxis],
                              img_4[..., np.newaxis],
                              img_5[..., np.newaxis],
                              img_6[..., np.newaxis],
                              img_7[..., np.newaxis]), axis=-1)
        img = torch.from_numpy(img).permute(3, 0, 1, 2)
        
        return (img[0].unsqueeze(0), img[1].unsqueeze(0), img[2].unsqueeze(0),
                img[3].unsqueeze(0), img[4].unsqueeze(0), img[5].unsqueeze(0),
                img[6].unsqueeze(0), ID)

    def __len__(self):
        return self.len


def run_task_inference(task, data_dir, val_split_path, bs, epoch, seed, num_class=2):
    """Run inference for a single task (S1 or S4)"""
    # Auto-generate model path
    model_path = f'./trained_models/trained_models_{task}/bs{bs}_epoch{epoch}_seed{seed}/best_ACC_val.pth'
    
    # Create temporary save directory
    save_dir = f'./results/val_results/val_results_{task}'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f'\nProcessing {task} task...')
    print(f'Using model: {model_path}')

    # Load dataset
    dataset = Dataset_for_external_validation(data_dir, val_split_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    print(f'Dataset length for {task}: {len(dataset)}')

    # Initialize model
    net = ResNet18_3D_7stream_LSTM(in_channels=1, n_classes=num_class, pretrained=False, no_cuda=False)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for {task} inference")
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    # Check model existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # Run inference
    predictions = []
    prob_column = f'Subtask1_prob_{task}' if task == 'S1' else f'Subtask1_prob_{task}'
    
    with torch.no_grad():
        for i, (img1, img2, img3, img4, img5, img6, img7, case_ids) in enumerate(dataloader):
            # Move to GPU
            imgs = [img1, img2, img3, img4, img5, img6, img7]
            imgs = [x.cuda().float() for x in imgs]

            # Forward pass
            outputs = net(*imgs)
            outputs = torch.softmax(outputs, dim=1)
            prob_class_1 = outputs[:, 1].cpu().numpy()

            # Store results
            for j, case_id in enumerate(case_ids):
                predictions.append({
                    'Case': case_id,
                    'Setting': 'Contrast',
                    prob_column: prob_class_1[j]
                })
            
            if (i + 1) % 10 == 0:  # Print progress every 10 batches
                print(f'{task} processed {i + 1}/{len(dataloader)} cases')

    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    print(f'{task} inference completed. Total cases: {len(df)}')
    return df


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run S1 and S4 validation and merge results')
    parser.add_argument('--output_csv', type=str, default='./results/LiFS_pred.csv',
                      help='Path to save merged output CSV')
    parser.add_argument('--bs', type=int, default=4, help='Batch size used in training')
    parser.add_argument('--epoch', type=int, default=200, help='Epochs used in training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in training')
    args = parser.parse_args()

    # Common configurations
    data_dir = "./data/preprocessed_val/"
    val_split_path = './relevant_files/val.txt'

    # Run both tasks
    df_s1 = run_task_inference('S1', data_dir, val_split_path, args.bs, args.epoch, args.seed)
    df_s4 = run_task_inference('S4', data_dir, val_split_path, args.bs, args.epoch, args.seed)

    # Process S1 results (rename column)
    df_s1 = df_s1[["Case", "Subtask1_prob_S1"]].rename(
        columns={"Subtask1_prob_S1": "Subtask2_prob_S1"}
    )

    # Merge results
    print('\nMerging results from S1 and S4 tasks...')
    df_merged = pd.merge(
        df_s4[["Case", "Setting", "Subtask1_prob_S4"]],
        df_s1,
        on="Case",
        how="inner"
    )

    # Reorder columns
    df_merged = df_merged[["Case", "Setting", "Subtask1_prob_S4", "Subtask2_prob_S1"]]

    # Save merged results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_merged.to_csv(args.output_csv, index=False)

    # Print verification
    print(f'\nMerged results saved to: {args.output_csv}')
    print('\nFirst 5 rows of merged results:')
    print(df_merged.head())

    print('\nSummary statistics:')
    print(f'Total merged cases: {len(df_merged)}')
    print(f'Subtask1_prob_S4 - Mean: {df_merged["Subtask1_prob_S4"].mean():.4f}, '
          f'Std: {df_merged["Subtask1_prob_S4"].std():.4f}')
    print(f'Subtask2_prob_S1 - Mean: {df_merged["Subtask2_prob_S1"].mean():.4f}, '
          f'Std: {df_merged["Subtask2_prob_S1"].std():.4f}')


if __name__ == "__main__":
    main()