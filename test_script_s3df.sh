#!/bin/bash
#SBATCH --partition=ampere
## SBATCH --account=lcls:prjs2e21
#SBATCH --job-name=reg
#SBATCH --output=/sdf/data/lcls/ds/prj/prjs2e21/results/DCNS_LSTM_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --error=/sdf/data/lcls/ds/prj/prjs2e21/results/DCNS_LSTM_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-1:00:00
#SBATCH --gpus 1
source ~/conda.sh

echo starting run 1 at: `date`
# Check which GPU is being used
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Run the Python script with the specified arguments
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

#python3 main.py --model LSTM --data_dir /fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data --output_dir "/sdf/scratch/lcls/ds/prj/prjs2e21/scratch/dcns_lstm_output/test_06032024" --custom_code 1 --batch_size 200 --load_in_gpu 0

python3 main.py --model LSTM --data_dir /fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data --do_analysis 1 --model_param_path /fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data/LSTM_120_epoch_44.pth --analysis_file 98 --analysis_example 25 --output_dir /sdf/data/lcls/ds/prj/prjs2e21/results/DCNS_LSTM_Output/ --fig_save_dir /sdf/data/lcls/ds/prj/prjs2e21/results/DCNS_LSTM_Output/
#93 56, 98 25		
#python3 main.py --model LSTM --data_dir /fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data --do_prediction 1 --model_param_path /fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data/LSTM_120_epoch_44.pth --output_dir /sdf/data/lcls/ds/prj/prjs2e21/results/DCNS_LSTM_Output/
echo Finished at: `date`
