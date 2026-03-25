#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

#SBATCH --array=1-4

# --- 1. SET CUDA ENVIRONMENT ---
module load cuda/12.4.1 2>/dev/null
AUTO_CUDA=$(dirname $(dirname $(which nvcc)) 2>/dev/null)
KUMA_DEFAULT="/ssoft/spack/pinot-noir/kuma-h100/v2/spack/opt/spack/linux-rhel9-zen4/gcc-13.2.0/cuda-12.4.1-4h4wdshlayz4fomm4m5z2euvtjuhyoyy"

export CUDA_HOME=${AUTO_CUDA:-$KUMA_DEFAULT}
echo "🚀 Using CUDA_HOME: $CUDA_HOME"

# --- 2. LIBRARY LINKING ---
VENV_PATH="/home/dittrich/venvs/kandel"
VENV_CUDNN="$VENV_PATH/lib/python3.11/site-packages/nvidia/cudnn/lib"

export LD_LIBRARY_PATH=$VENV_CUDNN:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- 3. JAX / XLA / MUJOCO CONFIG ---
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

# --- 4. EXECUTION ---
source $VENV_PATH/bin/activate

echo "🛠️ Starting Training: $(date)"
echo "📍 Node: $HOSTNAME"

# python scripts/train_ant_mutilated_kandel.py \
#     num_generations=500 \
#     strategy.population_size=256 \
#     agent_seed=${SLURM_ARRAY_TASK_ID} \
#     checkpoint_interval=10 \
#     repeats=2 \
#     checkpoint_directory=/home/dittrich/data/bonney \
#     wandb_project=bonney

# train_playground_kandel.py \
#     num_generations=500 \
#     playground=CheetahRun \
#     strategy.population_size=256 \
#     agent_seed=${SLURM_ARRAY_TASK_ID} \
#     checkpoint_interval=10 \
#     repeats=2 \
#     checkpoint_directory=/home/dittrich/data/bonney \
#     wandb_project=bonney

echo "✅ Training (${SLURM_JOBID}) finished: $(date)"
