#!/bin/bash
#SBATCH --account=AIFIN_WP4
#SBATCH --error=test_llama.err
#SBATCH --output=test_llama.out
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=test_llama
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4

# Load modules
module purge
module load cuda
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gcc/12.2.0

# Get master node info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master node: $head_node"
echo "Master node IP: $head_node_ip"

# NCCL settings for Leonardo (InfiniBand)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0             
export NCCL_P2P_DISABLE=0             
export NCCL_SOCKET_IFNAME=ib0         
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_QPS_PER_CONNECTION=4

# General settings
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

CONFIG_FILE=${CONFIG_FILE:-"/leonardo_work/PHD_baroncel/torchtitan-0.2.2/torchtitan/models/llama3/train_configs/llama3_8b_4nn.toml"}


# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun torchrun --nnodes 4 --nproc_per_node 4 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"

