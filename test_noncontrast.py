import os
import shutil
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from ResNet_3D import ResNet18_3D_3stream_LSTM
from Nii_utils import NiiDataRead
import argparse


class Dataset_for_external_validation(Dataset):
    def __init__(self, data_dir, split_path):
        self.data_dir = data_dir
        with open(split_path, 'r') as f:
            ID_list_original = f.readlines()
        self.ID_list = [n.strip('\n') for n in ID_list_original]
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        img_1, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '1_img.nii.gz'))
        img_2, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '2_img.nii.gz'))
        img_3, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '3_img.nii.gz'))

        img = np.concatenate((img_1[..., np.newaxis],
                              img_2[..., np.newaxis],
                              img_3[..., np.newaxis]), axis=-1)
        img = torch.from_numpy(img).permute(3, 0, 1, 2)
        img_1 = img[0].unsqueeze(0)
        img_2 = img[1].unsqueeze(0)
        img_3 = img[2].unsqueeze(0)

        return img_1, img_2, img_3, ID

    def __len__(self):
        return self.len


def run_task(task, args):
    data_dir = "./data/preprocessed_val/"
    val_split_path = './relevant_files/val.txt'
    num_class = 2

    # Generate model path automatically if not specified
    if args.model_path is None:
        model_path = f'./noncontrast_trained_models/trained_models_{task}/bs{args.bs}_epoch{args.epoch}_seed{args.seed}/best_ACC_val.pth'
    else:
        model_path = args.model_path.replace('S1', task).replace('S4', task)

    # Generate save directory automatically if not specified
    if args.save_dir is None:
        save_dir = f'./noncontrast_results/val_results/val_results_{task}'
    else:
        save_dir = args.save_dir.replace('S1', task).replace('S4', task)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f'\nProcessing {task} task...')
    print(f'Results will be saved to: {save_dir}')
    print(f'Using model path: {model_path}')

    # Load dataset
    exval_data = Dataset_for_external_validation(data_dir, val_split_path)
    exval_dataloader = DataLoader(dataset=exval_data, batch_size=1, shuffle=False, drop_last=False)
    print(f'Dataset length: {exval_data.len}')

    # Initialize model
    net = ResNet18_3D_3stream_LSTM(in_channels=1, n_classes=num_class, pretrained=False, no_cuda=False)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_path}\n"
            f"Check task ({task}) or training parameters (bs={args.bs}, epoch={args.epoch}, seed={args.seed})"
        )

    net.load_state_dict(torch.load(model_path))
    print(f'Successfully loaded model from: {model_path}')

    # Inference
    print('Starting prediction...')
    net.eval()
    predictions = []
    # Subtask numbering: S4 -> Subtask1, S1 -> Subtask2
    subtask_num = '2' if task == 'S1' else '1'
    prob_column = f'Subtask{subtask_num}_prob_{task}'

    with torch.no_grad():
        for i, (T1W1_imgs, T2W2_imgs, DWI_imgs, case_ids) in enumerate(exval_dataloader):
            T1W1_imgs = T1W1_imgs.cuda().float()
            T2W2_imgs = T2W2_imgs.cuda().float()
            DWI_imgs = DWI_imgs.cuda().float()

            outputs = net(T1W1_imgs, T2W2_imgs, DWI_imgs)
            outputs = torch.softmax(outputs, dim=1)
            prob_class_1 = outputs[:, 1].cpu().numpy()

            for j, case_id in enumerate(case_ids):
                predictions.append({
                    'Case': case_id,
                    'Setting': 'NonContrast',
                    prob_column: prob_class_1[j]
                })

            print(f'Processed {i + 1}/{len(exval_dataloader)} cases')

    # Save task-specific results
    if args.output_csv is None:
        output_csv = f'val_predictions_{task}.csv'
    else:
        output_csv = args.output_csv.replace('S1', task).replace('S4', task)
    output_path = os.path.join(save_dir, output_csv)
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)

    print(f'\nTask {task} completed')
    print(f'Total cases processed: {len(predictions)}')
    print(f'Results saved to: {output_path}')
    return output_path


def merge_results(s1_path, s4_path, output_path):
    print("\nMerging results...")
    
    # Load S4 results (Subtask1)
    df_s4 = pd.read_csv(s4_path)
    df_s4 = df_s4[["Case", "Setting", "Subtask1_prob_S4"]]

    # Load S1 results (Subtask2)
    df_s1 = pd.read_csv(s1_path)
    df_s1 = df_s1[["Case", "Subtask2_prob_S1"]]

    # Merge results
    df_merged = pd.merge(df_s4, df_s1, on="Case", how="inner")
    df_merged = df_merged[["Case", "Setting", "Subtask1_prob_S4", "Subtask2_prob_S1"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    
    print(f"Merged CSV saved to: {output_path}")
    print("\nFirst 5 rows of merged CSV:")
    print(df_merged.head())


def main():
    parser = argparse.ArgumentParser(description='Run S1 and S4 validation and merge results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model checkpoint (auto-generated if not specified)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save output CSV file (auto-generated if not specified)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save validation results (auto-generated if not specified)')
    parser.add_argument('--merged_output', type=str, default="./noncontrast_results/LiFS_pred.csv",
                        help='Path to save merged CSV file')
    parser.add_argument('--bs', type=int, default=4, help='Batch size used in model training')
    parser.add_argument('--epoch', type=int, default=200, help='Epochs used in model training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in model training')
    args = parser.parse_args()

    # Run both tasks sequentially
    s1_csv = run_task('S1', args)
    s4_csv = run_task('S4', args)

    # Merge results from two tasks
    merge_results(s1_csv, s4_csv, args.merged_output)

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()