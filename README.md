## Usage

### 1. Download Data
Download the training dataset `LiQA_training_data.zip` and validation dataset `LiQA_val.zip`, then unzip them.


### 2. Environment Setup
We recommend installing dependencies via the following version-locked commands (all versions are verified and reproducible). Run the commands sequentially:
```bash
# Create and activate the conda environment
conda create -n 34lifs python=3.10 -y
conda activate 34lifs

# Install PyTorch, CUDA, and related packages via conda (fixed version)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install data processing & image-related packages via conda (verified fixed versions)
conda install pandas==2.3.3 scikit-learn==1.7.2 scikit-image==0.25.2 SimpleITK==2.2.1 -y

# Install remaining dependencies via pip (verified fixed versions)
pip install tensorboardX==2.6.4
pip3 install opencv-python==4.8.1.78 --force-reinstall
pip3 install matplotlib==3.10.7
pip3 install numpy==1.26.4 --force-reinstall
```


### 3. Data Preprocessing
1. Copy the unzipped `LiQA_training_data` and `LiQA_val` to the designated path and rename them. Refer to `./data/train_data.py` and `./data/val_data.py`, replace `SOURCE_ROOT` with the actual input data path. Then run the following commands to generate raw processed data:
   ```bash
   cd 34-LiFS
   cd ./data
   python train_data.py
   python val_data.py
   cd ..
   ```
   The raw output data will be saved in `./data/train/` and `./data/val/`.

2. Preprocess the training and validation datasets by running the following commands:
   ```bash
   python preprocessing_train.py
   python preprocessing_val.py
   ```

3. The preprocessed training and validation datasets will be saved in `./data/preprocessed_train/` and `./data/preprocessed_val/` respectively.


### 4. Model Training
#### 4.1 Train the Substantial Fibrosis Detection Model (S1 vs. S2–S4)
Run the following command to train the classification model for Substantial Fibrosis Detection (S1 vs. S2–S4):
```bash
python train.py
```

#### 4.2 Train the Cirrhosis Detection Model (S1–S3 vs. S4)
After the above training is completed, modify `S1` to `S4` in `metadata_path` and `save_dir` in `train.py`, then run the same command again to train the classification model for Cirrhosis Detection (S1–S3 vs. S4).

#### 4.3 Alternative: Use Pre-trained Models
Alternatively, you can download the two pre-trained model zip files via the following **Baidu Netdisk link**, then place them in the `./trained_models/` path and unzip them.
- Baidu Netdisk Link: https://pan.baidu.com/s/1ZD-c-vWxR85_3QTNXWefvA
- Extraction Code: 5ch3


### 5. Model Testing
1. Prepare `val.txt` in the `relevant_files` folder (this file contains Case ID information).
   - If you use other test sets, refer to `./data/val_data.py` and `preprocessing_val.py` to process the test set, save it in a new folder, and prepare the corresponding `.txt` file.

2. Ensure `data_dir` and `val_split_path` in `test.py` are correctly configured. If other test sets are used, further modify the output path (e.g., create a new `test_results` folder in `./results/` and update the path accordingly).

3. Run the following command:
   ```bash
   python test.py
   ```

4. Output results are saved in the `results` folder.


### 6. Non-Contrast Subtask
For the Non-Contrast Subtask, you only need to replace `train.py` with `train_noncontrast.py` and `test.py` with `test_noncontrast.py` in all training/testing commands.

The two pre-trained model zip files for the Non-Contrast Subtask can also be downloaded via the same Baidu Netdisk link provided above.
