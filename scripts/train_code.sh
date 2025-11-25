#!/bin/bash
#
CWD=`realpath -s $0`
CWD=`dirname ${CWD}`
PWD=`dirname ${CWD}`

model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

dataset="code"
max_prompt_length=1024
max_response_length=2048
hint_lower_prob=0.05
hint_upper_prob=0.95
hint_sampling_uniform="False"

total_training_steps=500
total_training_steps_with_hint=300
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=4
hint_sft_loss_coef=0.001
ref_rollout_n=4

nnodes=1
n_gpus_per_node=8
seed=0

alg="xft"

help() {
    echo "usage: `basename $0` [-h] -m <model> -d <dataset> -- [train_opt ...]"
    echo "options:"
    echo "    -h, --help show this help message and exit"
    echo "    -m MODEL, --model MODEL"
    echo "               base model. (default: \"${model}\")"
    echo "    -d DATASET, --dataset DATASET"
    echo "               dataset. (default: \"${dataset}\")"
    echo "        --max_prompt_length MAX_PROMPT_LENGTH"
    echo "                           max prompt length. (default: ${max_prompt_length})"
    echo "        --max_response_length MAX_RESPONSE_LENGTH"
    echo "               max response length. (default: ${max_response_length})"
    echo "        --hint_lower_prob HINT_LOWER_PROB"
    echo "               hint lower prob. (default: ${hint_lower_prob})"
    echo "        --hint_upper_prob HINT_UPPER_PROB"
    echo "               hint lower prob. (default: ${hint_upper_prob})"
    echo "        --hint_sampling_uniform uniform sampling hint. (default: ${hint_sampling_uniform})"
    echo "    -t STEPS, --total_training_steps STEPS"
    echo "               total training steps. (default: ${total_training_steps})"
    echo "        --total_training_steps_with_hint STEPS"
    echo "               total training steps with hint. (default: ${total_training_steps_with_hint})"
    echo "        --ppo_mini_batch_size BATCH_SIZE"
    echo "               ppo_mini_batch_size. (default: ${ppo_mini_batch_size})"
    echo "        --ppo_micro_batch_size_per_gpu BATCH_SIZE"
    echo "               ppo_micro_batch_size_per_gpu. (default: ${ppo_micro_batch_size_per_gpu})"
    echo "        --hint_sft_loss_coef SFT_LOSS_COEF"
    echo "               sft loss coef for hint. (default: ${hint_sft_loss_coef})"
    echo "        --ref_rollout_n ROLLOUT_N"
    echo "               rollout_ref.rollout_n. (default: ${ref_rollout_n})"
    echo "    -n NNODES, --nnodes NNODES"
    echo "               number of nodes. (default: ${nnodes})"
    echo "        --n_gpus_per_node NGPUS_PER_NODE"
    echo "               number of gpus per node. (default: ${n_gpus_per_node})"
    echo "        --seed SEED"
    echo "               seed everything. (default: ${seed})"
    exit $1
}

ARGS=$( \
    getopt \
        -o "m:d:t:n:h" \
	-l "model:,
	    dataset:,
	        max_prompt_length:,
		max_response_length:,
		hint_lower_prob:,
		hint_upper_prob:,
		hint_sampling_uniform,
	    total_training_steps:,
	        total_training_steps_with_hint:,
		ppo_mini_batch_size:,
		ppo_micro_batch_size_per_gpu:,
		hint_sft_loss_coef:,
	    nnodes:,
	        n_gpus_per_node:,
	        seed:,
	    help" \
	-- "$@" \
) || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-m | --model) model="$2"; shift 2;;
    (-d | --dataset) dataset="$2"; shift 2;;
    (--max_prompt_length) max_prompt_length="$2"; shift 2;;
    (--max_response_length) max_response_length="$2"; shift 2;;
    (--hint_lower_prob) hint_lower_prob="$2"; shift 2;;
    (--hint_upper_prob) hint_upper_prob="$2"; shift 2;;
    (--hint_sampling_uniform) hint_sampling_uniform="True"; shift 1;;
    (-t | --total_training_steps) total_training_steps="$2"; shift 2;;
    (--total_training_steps_with_hint) total_training_steps_with_hint="$2"; shift 2;;
    (--ppo_mini_batch_size) ppo_mini_batch_size="$2"; shift 2;;
    (--ppo_micro_batch_size_per_gpu) ppo_micro_batch_size_per_gpu="$2"; shift 2;;
    (--hint_sft_loss_coef) hint_sft_loss_coef="$2"; shift 2;;
    (-n | --nnodes) nnodes="$2"; shift 2;;
    (--n_gpus_per_node) n_gpus_per_node="$2"; shift 2;;
    (--seed) seed="$2"; shift 2;;
    (-h | --help) help 0;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

EXPERIMENT_NAME="${dataset}-${model}-${alg}"
default_local_dir=${PWD}/checkpoints/"${dataset}-${model}-${alg}-${max_prompt_length}-${max_response_length}"
VLLM_ATTENTION_BACKEND="XFORMERS"

data_dir="${PWD}/data/${dataset}"

export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONPATH=verl:. python -m xft.trainer.main_ppo \
    data.train_files=${data_dir}/train.parquet \
    data.val_files=${data_dir}/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation="right" \
    custom_reward_function.path=xft/utils/reward_score/reward_score.py \
    custom_reward_function.name=reward_func \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    +actor_rollout_ref.actor.sft_loss_coef=${hint_sft_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ref_rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path=${model} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    +trainer.seed=${seed} \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.project_name=xft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir=$default_local_dir \
    +data.trainer.total_training_steps_hint=${total_training_steps_with_hint} \
    +data.trainer.uniform_sampling=${hint_sampling_uniform} \
    +data.trainer.stage=False \
    +data.trainer.lower_prob=${hint_lower_prob} \
    +data.trainer.upper_prob=${hint_upper_prob} \
    +data.trainer.split_prompt=True \
    trainer.total_epochs=30000 \
    $*
