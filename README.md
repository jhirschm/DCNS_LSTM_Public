# DCNS_LSTM_Public
Data Preprocessing, LSTM Model, and Analysis for an LSTM for Solving a Complex NLO Process


This document provides a comprehensive guide to using the LSTM model training and analysis script. It details the command line arguments available for configuring the model, datasets, and the training process.

## Overview
The script allows training, testing, and analysis of LSTM-based neural networks for solving a complex NLO process. It supports multiple functionalities, including hyperparameter tuning, model prediction, data analysis, and custom loss functions.

## Requirements
Ensure that your Python environment is correctly set up with necessary dependencies which are listed in the `requirements.txt` file. You can install them using the following command:

```
pip install -r requirements.txt
```

## Usage
Run the script from the command line with the required options. Here is the basic command structure:

```bash
python main.py --model <MODEL_TYPE> --data_dir <PATH_TO_DATA>
```

### Command Line Arguments
Below is a list of all supported command line arguments along with their descriptions:

- `--model`: Specifies the type of model to use for training. This argument is required. The supported values are `LSTM`, `BlindTridentLSTM`, and `FFLSTM`.
- `--model_save_name`: Sets the name under which the model will be saved. Defaults to `LSTM_model_latest`.
- `--data_dir`: Path to the directory containing the dataset. This argument is required.
- `--num_epochs`: Number of epochs for training. Defaults to 1.
- `--custom_loss`: Name of the custom loss function to use. Defaults to `MSE`.
- `--shg_weight`: Weight for the SHG component of the loss. Optional and only used if your custom loss function requires it.
- `--sfg_weight`: Weight for the SFG component of the loss. Optional and only used if your custom loss function requires it.
- `--epoch_save_interval`: Interval for saving model checkpoints during training. Defaults to every epoch.
- `--tune_train`: Set to 1 to enable hyperparameter tuning. Defaults to 0 (disabled).
- `--output_dir`: Directory for saving output files. Defaults to the current directory where the script is run.
- `--do_prediction`: Set to 1 to enable prediction mode. Defaults to 0 (disabled). The prediction mode requires the `model_param_path` argument and what it will do is load the model parameters from the specified path and run an entire inference on the test dataset (although it can be used for any dataset). It will try to emulate an entire crystal by running the model on each slice of the crystal and then concatenating the results to create output and the input for the next slice. Note that this will also run the analysis code on the test dataset.
- `--do_analysis`: Set to 1 to perform analysis after training or prediction. Defaults to 0 (disabled). Analysis here means running the model on a specific example and plotting the results in a human-readable format, such as the ones in the main paper.
- `--model_param_path`: Path to pre-trained model parameters for initialization or analysis. If you pass this and try to train the model, it will load the model parameters from this path and continue training from there. If you pass this and try to predict or analysis, it will load the model parameters from this path and run the prediction or analysis on the test dataset.
- `--verbose`: Enable verbose output during operations. Defaults to 1 (enabled).
- `--batch_size`: Batch size for training and prediction. Aim to maximize GPU memory usage. Defaults to 7500.
- `--analysis_file`: Index of the file used for final analysis. Defaults to 91.
- `--analysis_example`: Example index within the analysis file. Defaults to 15.
- `--crystal_length`: Assumed length of the crystal being modeled. Defaults to 100 as per our dataset.
- `--is_slice`: Whether modeling a slice of the crystal (1) by the neural net or the entire crystal at once (0). Defaults to 1.
- `--lstm_hidden_size`: Hidden layer size of the LSTM. Defaults to 1024.
- `--lstm_num_layers`: Number of layers in the LSTM. Defaults to 1.
- `--lstm_linear_layer_size`: Size of the linear layers following the LSTM. Defaults to 4096.
- `--loss_reduction`: Type of reduction to apply in the loss function. Defaults to `mean`.
- `--lr`: The initial learning rate for the optimizer. Defaults to 0.0001.
- `--lstm_dropout`: Dropout rate for the LSTM layers. Defaults to 0.3.
- `--fc_dropout`: Dropout rate for the fully connected layers after LSTM. Defaults to 0.3.
- `--train_load_mode`: Loading mode for the training dataset. Defaults to 0.
- `--val_load_mode`: Loading mode for the validation dataset. Defaults to 0.
- `--test_load_mode`: Loading mode for the test dataset. Defaults to 1.
- `--shuffle`: Whether to shuffle the training dataset. Defaults to 1 (enabled).
- `--has_fc_dropout`: Whether to include dropout layers after LSTM. Defaults to 1 (enabled).
- `--bidirectional`: Whether the LSTM should be bidirectional. Defaults to 0 (disabled).
- `--custom_code`: Used for running codes regarding the timing and final evaluation metric through the main function. Defaults to 0 which means doing nothing special. It takes in either 0, 1, 2, or 3. With 1, it will run the timing function for a previous version of the model. With 2, it will run the timing function for the current version of the model. With 3, it will run the final evaluation metric code.
- `--load_in_gpu`: Whether to load the data directly into GPU memory. Defaults to 1 (enabled).
- `--cpu_cores`: Number of CPU cores to utilize. Optional.

## Examples
To train a model with specific parameters:
```bash
python main.py --model LSTM --data_dir /path/to/data --num_epochs 10 --lr 0.001 --batch_size 512 --verbose 1
```

For prediction only:
```bash
python main.py --model LSTM --do_prediction 1 --model_param_path /path/to/model.pth --data_dir /path/to/data
```

For analysis only:
```bash
python main.py --model LSTM --do_analysis 1 --model_param_path /path/to/model.pth --data_dir /path/to/data --analysis_file 91 --analysis_example 15
```

For timing analysis:
```bash
python3 main.py --model LSTM --data_dir /path/to/data --custom_code 2
```

For generating the histogram of the final evaluation metric:
```bash
python3 main.py --model LSTM --data_dir /path/to/data --output_dir "/mnt/twoterra/outputs/14-02-2024/" --model_save_name EXAMPLE_MODEL --custom_code 3
```


## Support
For any issues or additional questions, please refer to the comments in the code.