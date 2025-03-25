import numpy as np
import h5py 
import os
import matplotlib.pyplot as plt

data_dir = "/fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data"
file_idx = 1
def get_data_labels(file_idx, sample_idx):
    with h5py.File(os.path.join(data_dir, "X_new_data.h5"), "r") as file:
        x_dataset = file[f"dataset_{file_idx}"]

        

        data = x_dataset[sample_idx]

    with h5py.File(os.path.join(data_dir, "y_new_data.h5"), "r") as file:
        y_dataset = file[f"dataset_{file_idx}"]
        # If training or funky analysis, we want to load the entire dataset
        
        labels = y_dataset[sample_idx]
    return data, labels
data0, labels0 = get_data_labels(0,0)
data1, labels1 = get_data_labels(0,1)
data2, labels2 = get_data_labels(0,2)
data99, labels99 = get_data_labels(0,99)


    