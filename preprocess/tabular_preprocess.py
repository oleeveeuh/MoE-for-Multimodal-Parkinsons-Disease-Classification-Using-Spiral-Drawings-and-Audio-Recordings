import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
import shutil


#filter for only lines recording Static Spiral Test
#remove Grip Angle, TestID

def trim_tabular(input_file, output_dir):
    """
    Slice a WAV audio file into segments and concatenate them in specified groups. If there are not enough segments,
    concatenate all valid segments.

    :param input_file: Path to the input WAV audio file (e.g., "his_one/input.wav")
    :param output_dir: Directory path to save the output files (e.g., "./output")
    :param segments_to_concat: Number of segments to concatenate each time, default is 5
    :return: Updated counter for the output file names
    """
    features = []


    with open(file_path, 'r') as file:
        lines = file.readlines()

    first_timestamp = int((lines[0].split(';'))[5])

    for line in lines:
            # Strip whitespace and split by ';'
        data = line.strip().split(';')

            # Check if Test ID is '0' before processing the line
        test_id = data[6].strip()
        if test_id != '0':
            continue  # Skip this line if Test ID is not 0

            # Parse values from the line
        x = int(data[0].strip())   # X coordinate
        y = int(data[1].strip())   # Y coordinate
        z = int(data[2].strip())   # Z coordinate
        pressure = int(data[3].strip())  # Pressure
        grip_angle = int(data[4].strip())  # GripAngle (not used in calculations here)
        timestamp = int(data[5].strip()) - first_timestamp # Timestamp

        features.append([x, y, z, pressure, timestamp])

    features = torch.tensor(features, dtype=torch.float32)
    np.savetxt(output_dir, features, delimiter=';', fmt='%.4f')

    print(f"Trimmed successfully, results located at: {output_dir}")


processed_txt = []
labels = []
path = "./Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/alldata/"
save_path = "./preprocessed_data/tabular/"

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    new_save_path = os.path.join(save_path, file)
    if file_path.endswith('.txt'):
        processed_txt.append(new_save_path)

        if file[0] == 'H':
            labels.append(0)
        else: labels.append(1)
        trim_tabular(file_path, new_save_path)

X_train, X_test, y_train, y_test = train_test_split(processed_txt, labels, test_size=0.2, random_state=42, stratify=labels)


for path in X_train:
    new_save_path = "./preprocessed_data/tabular/train"
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    shutil.copy(path,new_save_path)


for path in X_test:
    new_save_path = "./preprocessed_data/tabular/test"
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    shutil.copy(path,new_save_path)