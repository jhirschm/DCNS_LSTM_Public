import numpy as np
import h5py 
import os
import matplotlib.pyplot as plt

data_dir = "/fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data"
plot_path = "/sdf/home/j/jhirschm/Publications/DCNS_LSTM"
plot_name = "data_proc_fig1.pdf"
plot_save = os.path.join(plot_path, plot_name)
def get_data_labels(file_idx, sample_idx):
    with h5py.File(os.path.join(data_dir, "X_new_data.h5"), "r") as file:
        x_dataset = file[f"dataset_{file_idx}"]

        

        data = x_dataset[sample_idx]

    with h5py.File(os.path.join(data_dir, "y_new_data.h5"), "r") as file:
        y_dataset = file[f"dataset_{file_idx}"]
        # If training or funky analysis, we want to load the entire dataset
        
        labels = y_dataset[sample_idx]
    return data, labels

# Fetch required datasets
samples = [0, 1, 2, 99]
all_data_labels = [get_data_labels(0, idx) for idx in samples]

# Create plot
fig, axes = plt.subplots(nrows=4, ncols=11, figsize=(22, 8))

for row_idx, (data, labels) in enumerate(all_data_labels):
    # Plot first 10 data series
    for col_idx in range(10):
        axes[row_idx, col_idx].plot(data[col_idx])
        axes[row_idx, col_idx].axis('off')  # remove axis labels, ticks, etc.

    # Plot labels in 11th subplot
    axes[row_idx, 10].plot(labels)
    axes[row_idx, 10].axis('off')

plt.tight_layout()
plt.savefig(plot_save)
plt.show()


    