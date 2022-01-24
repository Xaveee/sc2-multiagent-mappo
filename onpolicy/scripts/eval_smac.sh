#!/bin/sh
env="StarCraft2"
map="8m"
algo="mappo"
exp="mlp"
user_name="xavee"
seed_max=1
model_dir="./results/StarCraft2/8m/mappo/mlp/wandb/run-20210921_170930-2n5y1gbd/files"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python eval/eval_smac.py --user_name ${user_name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 127 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --use_recurrent_policy --model_dir ${model_dir}
done
