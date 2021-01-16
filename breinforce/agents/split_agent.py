from breinforce.agents import BaseAgent


class SplitAgent(BaseAgent):
    """
    Agent who can do limited amount of fractual bets.
    """

    def __init__(self, actions, fracs):
        super().__init__(actions)
        self.fracs = fracs

    def act(self, action, frac, pot, call, min_raise, max_raise):
        bet = None
        if "fold" in action:
            bet = 0
        if "call" in action:
            bet = call
        if "raise" in action:
            bet = call + frac * pot
        if "allin" in action:
            bet = max_raise
        return bet
