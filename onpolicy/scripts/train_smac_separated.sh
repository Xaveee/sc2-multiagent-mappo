#!/bin/sh
# MAP LIST:
#     1c3s5z
#     2c_vs_64zg
#     2m_vs_1z
#     2s3z
#     2s_vs_1sz
#     3m
#     3s5z
#     3s5z_vs_3s6z
#     3s_vs_3z
#     3s_vs_4z
#     3s_vs_5z
#     5m_vs_6m
#     6h_vs_8z
#     8m
#     8m_vs_9m
#     10m_vs_11m
#     25m
#     27m_vs_30m
#     bane_vs_bane
#     corridor
#     MMM
#     MMM2
#     so many baneling
env="StarCraft2"
map="3s5z"
algo="mappo"
exp="agent_decentralized_1"
user_name="xavee"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --user_name ${user_name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 127 --n_rollout_threads 25 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --use_recurrent_policy --share_policy false --use_centralized_V false
done
