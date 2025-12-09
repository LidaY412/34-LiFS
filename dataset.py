from torch.utils.data import Dataset
import os
import torch
from Nii_utils import NiiDataRead
import pandas as pd
from volumentations import *
from sklearn.preprocessing import MinMaxScaler


class Dataset_for_tumor(Dataset):
    def __init__(self, data_dir, split_path, metadata_pre_path, augment=True):
        self.data_dir = data_dir
        self.augment = augment

        with open(split_path, 'r') as f:
            ID_list_orginal = f.readlines()
        ID_list_orginal = [n.strip('\n') for n in ID_list_orginal]

        metadata_df = pd.read_csv(metadata_pre_path)
        metadata_df['ID'] = metadata_df['ID'].astype(str)
        metadata_df['label'] = metadata_df['label'].astype(int)

        self.ID_list = []
        self.label_list = []

        for ID in ID_list_orginal:
            self.ID_list.append(ID)
            self.label_list.append(metadata_df.loc[metadata_df.ID == ID, 'label'].values[0])

        self.num_0 = self.label_list.count(0)
        self.num_1 = self.label_list.count(1)

        self.transforms = Compose([
            RotatePseudo2D(axes=(1, 2), limit=(-30, 30), interpolation=3, value=-1, p=0.3),
            ElasticTransformPseudo2D(alpha=50, sigma=30, alpha_affine=10, value=-1, p=0.3),
            GaussianNoise(var_limit=(0, 0.1), mean=0, p=0.3),
        ])

        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = torch.tensor(self.label_list[idx], dtype=torch.long)

        img_1, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '1_img.nii.gz'))
        img_2, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '2_img.nii.gz'))
        img_3, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '3_img.nii.gz'))
        img_4, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '4_img.nii.gz'))
        img_5, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '5_img.nii.gz'))
        img_6, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '6_img.nii.gz'))
        img_7, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '7_img.nii.gz'))
        img = np.concatenate((img_1[..., np.newaxis],
                              img_2[..., np.newaxis],
                              img_3[..., np.newaxis],
                              img_4[..., np.newaxis],
                              img_5[..., np.newaxis],
                              img_6[..., np.newaxis],
                              img_7[..., np.newaxis]), axis=-1)

        if self.augment:
            img = self.transforms(image=img)['image']

        img = torch.from_numpy(img).permute(3, 0, 1, 2)

        img_1 = img[0].unsqueeze(0)
        img_2 = img[1].unsqueeze(0)
        img_3 = img[2].unsqueeze(0)
        img_4 = img[3].unsqueeze(0)
        img_5 = img[4].unsqueeze(0)
        img_6 = img[5].unsqueeze(0)
        img_7 = img[6].unsqueeze(0)

        return img_1, img_2, img_3, img_4, img_5, img_6, img_7, label

    def __len__(self):
        return self.len


class Dataset(Dataset):
    def __init__(self, data_dir, split_path, metadata_pre_path, augment=True):
        self.data_dir = data_dir
        self.augment = augment

        with open(split_path, 'r') as f:
            ID_list_orginal = f.readlines()
        ID_list_orginal = [n.strip('\n') for n in ID_list_orginal]

        metadata_df = pd.read_csv(metadata_pre_path)
        metadata_df['ID'] = metadata_df['ID'].astype(str)
        metadata_df['label'] = metadata_df['label'].astype(int)

        self.ID_list = []
        self.label_list = []

        for ID in ID_list_orginal:
            self.ID_list.append(ID)
            self.label_list.append(metadata_df.loc[metadata_df.ID == ID, 'label'].values[0])

        self.num_0 = self.label_list.count(0)
        self.num_1 = self.label_list.count(1)

        self.transforms = Compose([
            RotatePseudo2D(axes=(1, 2), limit=(-30, 30), interpolation=3, value=-1, p=0.3),
            ElasticTransformPseudo2D(alpha=50, sigma=30, alpha_affine=10, value=-1, p=0.3),
            GaussianNoise(var_limit=(0, 0.1), mean=0, p=0.3),
        ])

        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = torch.tensor(self.label_list[idx], dtype=torch.long)

        img_1, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '1_img.nii.gz'))
        img_2, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '2_img.nii.gz'))
        img_3, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, '3_img.nii.gz'))

        img = np.concatenate((img_1[..., np.newaxis],
                              img_2[..., np.newaxis],
                              img_3[..., np.newaxis]), axis=-1)

        if self.augment:
            img = self.transforms(image=img)['image']

        img = torch.from_numpy(img).permute(3, 0, 1, 2)

        img_1 = img[0].unsqueeze(0)
        img_2 = img[1].unsqueeze(0)
        img_3 = img[2].unsqueeze(0)

        return img_1, img_2, img_3, label

    def __len__(self):
        return self.len
