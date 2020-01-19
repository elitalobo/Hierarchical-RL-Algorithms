My paper on Maximum-Entropy Option-Critic:
[shorturl.at/bkoL9](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/soc_report.pdf)

# Hierarchical-RL-Algorithms
Implemented the following papers in pytorch for solving Roboschool tasks: \
https://arxiv.org/pdf/1609.05140.pdf \
https://openreview.net/forum?id=Hyl_vjC5KQ \
Credits: https://github.com/TakaOsa/adInfoHRL \

Add soft-option critic which is an off-policy option-critic based on maximum entropy framework. Uses deep embedded clustering for clustering state-action space with ARC like feature representation for training different options. 


Improved AdinfoHRL by adding replacing the deterministic option-policies with stochastic option-policies and changing the objective to maximize entropy in a constrained manner as done in soft-actor-critic. \
The code structure and some parts of the code has been adopted from original adInfoHRL code .
**Run option critic:**  \
python hierarchical_dqn --env_name="HopperBulletEnv-v0" --options_cnt=4 

**Run adInfoHRL:** \
python adInfoHRLAlt.py --env_name="HopperBulletEnv-v0" --options_cnt=4 

**Run maximum entropy option-critic:** \
python maximum_entropy-option-critic/adinfoHRLAlt_with_kl_coeff.py --env_name="HopperBulletEnv-v0" --options_cnt=4 

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/maximum_entropy-option-critic/HalfCheetahBulletEnv-v0.png)

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/maximum_entropy-option-critic/HopperBulletEnv-v0.png)

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/maximum_entropy-option-critic/Walker2DBulletEnv-v0.png)

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/maximum_entropy-option-critic/oc-HalfCheetahBulletEnv-v0.png)

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/maximum_entropy-option-critic/oc-HopperBulletEnv-v0.png)

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/maximum_entropy-option-critic/oc-Walker2DBulletEnv-v0.png)

![alt text](https://github.com/elitalobo/Hierarchical-RL-Algorithms/blob/master/HopperBulletEnv-v0_state-action.png)

 `References`:
 [Learning Actionable Representations with Goal-Conditioned Policies](https://arxiv.org/pdf/1811.07819.pdf) \
 [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/pdf/1511.06335.pdf) \
 
 #TODO Add other references

