export ADV_ESTIMATOR=grpo
export USE_KL_LOSS="True"
export ROLLOUT_N=4
export SEED=7

data="math"
model="Qwen/Qwen2.5-1.5B"
alg="uft"

N_GPUS=8
#CUDA_VISIBLE_DEVICES=
BASE_MODEL=${model}
ROLLOUT_TP_SIZE=1
EXPERIMENT_NAME="${data}-${model}-${alg}"
VLLM_ATTENTION_BACKEND="XFORMERS"

LOWER_PROB=0.05
UPPER_PROB=0.95

DATA_DIR="./data/${data}"
TOTAL_TRAINING_STEPS=500
TOTAL_TRAINING_STEPS_HINT=300

PPO_MINIBATCH=64
PPO_MICROBATCH=4
ROLLOUT_LOGPROB_MICROBATCH=4
REF_LOGPROB_MICROBATCH=4
CRITIC_MICROBATCH=4

MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=1024
UNIFORM_SAMPLING="False"
SFT_LOSS_COEF=0.001

STAGE="False"

PYTHONPATH=verl:. python -m xft.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINIBATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICROBATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    +actor_rollout_ref.actor.sft_loss_coef=$SFT_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICROBATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICROBATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size_per_gpu=$CRITIC_MICROBATCH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    +trainer.seed=$SEED \
    trainer.logger='["console"]' \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.project_name=UFT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    +data.trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    +data.trainer.total_training_steps_hint=$TOTAL_TRAINING_STEPS_HINT \
    +data.trainer.uniform_sampling=$UNIFORM_SAMPLING \
    +data.trainer.stage=$STAGE \
    +data.trainer.lower_prob=$LOWER_PROB \
    +data.trainer.upper_prob=$UPPER_PROB \
    trainer.total_epochs=30000
