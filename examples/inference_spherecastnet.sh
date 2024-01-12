#!/bin/bash
# change the directory path of model run-time output and error messages to your own
#SBATCH --output=/scratch/gilbreth/gupt1075/infer_fourcastnet_nov.out
#SBATCH --error=/scratch/gilbreth/gupt1075/infer_fourcastnet_nov.err
# The file name of this submission file, so it's easier to track jobs
# filename: submit_run_model_example.sub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1 
#SBATCH --time=1:00:00
# partner queue has a 24-hour limit
#SBATCH -A gdsp-k|standby
#SBATCH -C  "v100|a100|a30"
# Job name, it will show up when you track this job
#SBATCH -J fourcastnet_job
# Use your email address so that you will receive email notifications about the job begin, end, or fail status
# To submit the job via command line:$  sbatch submit_run_model_example.sub 
# To check status of the submitted job:$  squeue -u yourUserID

module --force purge
unset PYTHONPATH
module load anaconda/5.3.1-py37
module load cuda/11.7.0
module load cudnn/cuda-11.7_8.6
module use /depot/gdsp/etc/modules
module load utilities monitor
module load rcac

module list
export PRECXX11ABI=1
export CUDA="11.7"

echo $PYTHONPATH

echo "$now"
echo "Current date completed loading modules: $now"



conda activate /home/gupt1075/.conda/envs/cent7/5.3.1-py37/pytorch


# Change this directory to where you save the model-related files such as run_model.py
# cd /scratch/gilbreth/wwtung/FourCastNet/

python /scratch/gilbreth/gupt1075/earth2mip/fcnv2_sm/simple_inference.py \
       --config="afno_backbone" \
       --run_num="02" \
       --fld="t850" \
       --weights="/scratch/gilbreth/gupt1075/model_weights/FCN_weights_v0/backbone.ckpt"  \
       --override_dir="/scratch/gilbreth/gupt1075/ERA5_expts_2m_temperature_trial_1/"






