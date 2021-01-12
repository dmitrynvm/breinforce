import gym
from breinforce import agents, envs

envs.Bropoker.configure()
env = gym.make('CustomSixPlayer-v0')
agents = [agents.RandomBaseAgent()] * 6
env.register_agents(agents)
obs = env.reset()

i = 0
while True:
    action = env.act(obs)
    obs, rewards, done, info = env.step(action)
    i += 1
    if all(done):
        break

print('Rewards', rewards)
