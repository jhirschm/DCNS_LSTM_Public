import numpy as np
import h5py 
import os
data_dir = "/fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data"
file_idx = 1
sample_idx = 1
with h5py.File(os.path.join(data_dir, "X_new_data.h5"), "r") as file:
    x_dataset = file[f"dataset_{file_idx}"]

    

    data = x_dataset[sample_idx]

with h5py.File(os.path.join(data_dir, "y_new_data.h5"), "r") as file:
    y_dataset = file[f"dataset_{file_idx}"]
    # If training or funky analysis, we want to load the entire dataset
    
    labels = y_dataset[sample_idx]

print(data.shape)
print(labels.shape)
    