#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=traffic
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:0
#SBATCH -p 'jsteinhardt'
#SBATCH -w 'balrog'
# #SBATCH -A fc_robustml
# #SBATCH -A co_stat

# set -x

# simulate conda activate flow
export PATH=/accounts/projects/jsteinhardt/aypan/value_learning:/accounts/projects/jsteinhardt/aypan/value_learning/flow:/accounts/projects/jsteinhardt/aypan/value_learning/finrl:/accounts/projects/jsteinhardt/aypan/sumo/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/envs/flow/bin:/accounts/projects/jsteinhardt/aypan/miniconda3/condabin:/usr/local/linux/anaconda3.8/bin:/accounts/projects/jsteinhardt/aypan/bin:/usr/local/linux/bin:/usr/local/bin:/usr/bin:/usr/sbin:/snap/bin:/usr/lib/rstudio-server/bin

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=$(($RANDOM + 1024))
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --redis-port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block & # --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
# __doc_head_ray_end__

# __doc_script_start__
CONFIG=$1
EXP=$2
NAME=$3
REWARD=$4
WEIGHT=$5
WIDTH=$6
DEPTH=$7

if [ "${CONFIG}" = "test" ]; then
    python3 -u traffic_proxy.py singleagent_merge_bus "test" delay,still 1,0.2 32 3 --n_cpus "$SLURM_CPUS_PER_TASK" --num_steps 10 --rollout_size 7 --horizon 300 --checkpoint 1 --test
    exit 0 
fi
if [ "${CONFIG}" = "ss" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK" --num_steps 10000 --rollout_size 7 --horizon 300 
elif [ "${CONFIG}" = "ls" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK" 
elif [ "${CONFIG}" = "sm" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK"  --num_steps 5000 --rollout_size 7 --horizon 300 --multi
elif [ "${CONFIG}" = "lm" ]; then
    python3 -u traffic_proxy.py ${EXP} ${NAME} ${REWARD} ${WEIGHT} ${WIDTH} ${DEPTH} --n_cpus "$SLURM_CPUS_PER_TASK" --multi 
else
    echo "Must select either 'ss' for short, single agent; 'ls' for long, single agent; 'sm' for short, multi agent; 'lm' for long, multi agent not ${CONFIG}"
    exit 0
fi 
