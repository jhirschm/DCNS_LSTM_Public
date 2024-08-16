#!/bin/bash
#SBATCH --partition=ampere
## SBATCH --account=lcls:prjs2e21
#SBATCH --job-name=reg
#SBATCH --output=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --error=/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/s3df_runtime_outputs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 4

python3 main.py --model LSTM --data_dir /fs/ddn/sdf/group/lcls/ds/scratch/s2e_scratch/Data/DCNS_NLO_LSTM_H5_Data --output_dir "/sdf/scratch/lcls/ds/prj/prjs2e21/scratch/dcns_lstm_output/test_06032024" --custom_code 1 --batch_size 200 --load_in_gpu 0
