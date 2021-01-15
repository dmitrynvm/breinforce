import gym
from breinforce import agents, envs, views
from breinforce.views import AsciiView, HandsView

envs.Bropoker.configure()
env = gym.make("CustomSixPlayer-v0")
ascii_view = AsciiView(env)
hands_view = HandsView(env)

actns = [
    "fold",
    "check",
    "call",
    "raise_half_pot",
    "raise_one_pot",
    "raise_two_pot",
    "allin"
]
fracs = [
    -float("inf"),
    0,
    None,
    0.5,
    1,
    2,
    float("inf")
]
probs = [
    0.1,
    0.1,
    0.5,
    0.1,
    0.1,
    0.1,
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
lines = ""
for step, item in enumerate(env.history):
    line = ""
    state, player, action, info = item
    line += f"street: {str(state['street']+1)}"
    line += ", "
    line += f"player: {player}"
    line += ", "
    line += f"min_raise: {state['min_raise']}"
    line += ", "
    line += f"max_raise: {state['max_raise']}"
    line += ", "
    line += f"action: {str(action).rjust(3)}"
    if state["street"] == 0:
        line += ", "
        line += f"ante: {state['antes'][0]}"
    line += f" -> pot: {state['pot']}"
    lines += line + "\n"
print(lines)
