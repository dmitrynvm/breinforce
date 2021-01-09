import gym
from breinforce import agents, envs, views

envs.configure()
recorder = envs.Recorder()
env = gym.make('CustomSixPlayer-v0')


actions = [
    'fold',
    'call',
    'raise_half_pot',
    'raise_one_pot',
    'raise_two_pot',
    'allin'
]
splits = [
    0,
    None,
    0.5,
    1,
    1,
    float('inf')
]
probs = [
    0.0,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2
]

agents = [agents.RandomSplitAgent(actions, splits, probs)] * 6
env.register_agents(agents)
obs = env.reset()
view = views.HandsView(env.config)

# view = views.AsciiView()
# player = self.player
# active = self.active
# allin = self.active * (self.stacks == 0)
# community_cards = self.community_cards
# button = self.button
# done = all(self.__done())
# hole_cards = self.hole_cards
# pot = self.pot
# payouts = self.__payouts()
# street_commits = self.street_commits
# stacks = self.stacks

# screen = {
#     'player': player,
#     'active': active,
#     'allin': allin,
#     'community_cards': community_cards,
#     'button': button,
#     'done': done,
#     'hole_cards': hole_cards,
#     'pot': pot,
#     'payouts': payouts,
#     'prev_action': None if not self.history else self.history[-1],
#     'street_commits': street_commits,
#     'stacks': stacks,
#     'num_players': self.num_players,
#     'num_hole_cards': self.num_hole_cards,
#     'num_community_cards': sum(self.num_community_cards)
# }
# return view.render(screen)


history = []
while True:
    bet = env.act(obs)
    print(env.render())
    hand = env.step(bet)
    obs, rewards, done, info = hand
    history.append(hand)
    if all(done):
        break


print(view.render(history))
