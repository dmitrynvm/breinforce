import gym
from pydux import create_store
from breinforce import views
from breinforce.core.types import Episode
from . import engine


class BropokerEnv(gym.Env):

    def __init__(self, config):
        self.store = create_store(
            engine.reducers.root,
            engine.creators.state(config)
        )
        self.config = config
        self.history = []
        self.agents = None

    def predict(self, obs):
        return self.players[self.state.player].predict(obs)

    def step(self, action):
        player = self.state.player
        self.store.dispatch(engine.creators.step(action))
        self.history += [Episode(self.state.deepcopy(), player, action)]
        return engine.actions.results(self.state)

    def register(self, players):
        self.players = players

    def render(self, mode='pokerstars'):
        out = None
        if mode == 'jsonify':
            out = views.jsonify.render(self.history)
        elif mode == 'poker888':
            out = views.poker888.render(self.history)
        elif mode == 'pokerstars':
            out = views.pokerstars.render(self.history)
        return out

    def reset(self):
        self.store.dispatch(engine.creators.reset())
        self.history = []
        return engine.actions.observe(self.state)

    @property
    def state(self):
        return self.store.get_state()
