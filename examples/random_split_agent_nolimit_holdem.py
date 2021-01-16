import gym
from breinforce import agents, envs, views
from breinforce.views import AsciiView, HandsView

envs.Bropoker.configure()
env = gym.make("CustomSixPlayer-v0")
ascii_view = AsciiView(env)
hands_view = HandsView(env)

probs = [
    0.1, # fold
    0.3, # call
    0.2, # half_pot_raise
    0.2, # one_pot_raise
    0.1, # two_pot_rais
    0.0  # all_in_raise
]

agents = [agents.RandomAgent(probs)] * 6
env.register_agents(agents)
obs = env.reset()

while True:
    action = env.act(obs)
    #print(ascii_view.render())
    obs, rewards, done, info = env.step(action)
    if all(done):
        break

lines = ""
lines += "[\n"
for step, item in enumerate(env.history):
    line = ""
    state, player, action, info = item
    line += "\t{\n"
    line += f"\t\tstep: {str(step+1)}, \n"
    line += f"\t\tstreet: {str(state['street']+1)}, \n"
    line += f"\t\tplayer: {player}, \n"
    line += f"\t\tcall: {state['call']}, \n"
    line += f"\t\traise: [{state['min_raise']}, {state['max_raise']}], \n"
    line += f"\t\taction: {str(action).rjust(3)} \n"
    line += f"\t\tpot: {state['pot']} \n"
    line += f"\t\tstacks: {state['stacks']} \n"
    line += f"\t\tacted: {state['acted']} \n"
    line += f"\t\tactive: {state['active']} \n"
    line += f"\t\tcommitted: {state['committed']} \n"
    line += "\t}"
    end = ",\n" if step < len(env.history) - 1 else "\n"
    lines += line + end
lines += "]\n"
print(lines)
