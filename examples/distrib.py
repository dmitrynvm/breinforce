import gym

import ray
from ray import tune


from breinforce import agents, envs

envs.configure()
env = gym.make('CartPole-v0')
# agents = [agents.RandomBaseAgent()] * 6
# env.register_agents(agents)
# obs = env.reset()

# i = 0
# while True:
#     action = env.act(obs)
#     obs, rewards, done, info = env.step(action)
#     print(obs['active'])
#     i += 1
#     if all(done):
#         break

# print('Rewards', rewards)



ray.init()

tune.run(
    "PPO",
    stop={
        "timesteps_total": 10000,
    },
    config={
        "env": "CustomSixPlayer-v0",
        "num_workers": 7,
        "env_config": {
            "unity_worker_id": 52
        },
        "train_batch_size": 500,
    },
    checkpoint_at_end=True,
)
