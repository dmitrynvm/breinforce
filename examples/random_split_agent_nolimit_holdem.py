import gym
from breinforce import agents, envs, views
from breinforce.views import AsciiView, HandsView

envs.Bropoker.configure()
env = gym.make('CustomSixPlayer-v0')
ascii_view = AsciiView(env)
hands_view = HandsView(env)

actns = [
    'fold',
    'call',
    'raise_half_pot',
    'raise_one_pot',
    'raise_two_pot',
    'allin'
]
fracs = [
    0,
    None,
    0.5,
    1,
    2,
    float('inf')
]
probs = [
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0
]

agents = [agents.RandomSplitAgent(actns, fracs, probs)] * 6
env.register_agents(agents)
obs = env.reset()

while True:
    action = env.act(obs)
    #print(ascii_view.render())
    hand = env.step(action)

    obs, rewards, done, info = hand
    if all(done):
        break

print(hands_view.render())
#for item in env.history:
#    print(item)
