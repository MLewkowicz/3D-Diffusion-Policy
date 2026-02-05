# Examples (single GPU):
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
#
# Examples (multi-GPU DDP):
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0,1,2,3 ddp
# bash scripts/train_policy.sh dp3 calvin 0322 0 0,1 ddp

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
use_ddp=${6:-}  # set to "ddp" to launch with torchrun (multi-GPU)

if [ -n "$use_ddp" ]; then
    n_gpus=$(echo "$gpu_id" | tr ',' '\n' | wc -l)
    echo -e "\033[33mDDP mode: using GPUs ${gpu_id} (${n_gpus} processes)\033[0m"
else
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ -n "$use_ddp" ]; then
    torchrun --nproc_per_node=${n_gpus} --standalone train.py --config-name=${config_name}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}
else
    python train.py --config-name=${config_name}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}
fi



                                