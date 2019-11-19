# Hierarchical-RL-Algorithms

Implemented the following papers in pytorch for solving Roboschool tasks: \
https://arxiv.org/pdf/1609.05140.pdf \
https://openreview.net/forum?id=Hyl_vjC5KQ \
Credits: https://github.com/TakaOsa/adInfoHRL \
Add soft-option critic which is an off-policy option-critic based on maximum entropy framework. Uses kmeans for clustering state-action space for assigning different options. 

Improved AdinfoHRL by adding replacing the deterministic option-policies with stochastic option-policies and changing the objective to maximize entropy in a constrained manner as done in soft-actor-critic. \
**Run option critic:**  \
python hierarchical_dqn --env_name="HopperBulletEnv-v0" --options_cnt=4 

**Run adInfoHRL:** \
python adInfoHRLAlt.py --env_name="HopperBulletEnv-v0" --options_cnt=4 

**Run soft-option-critic:** \
python soc2_kmeans.py --env_name="HopperBulletEnv-v0" --options_cnt=4 

