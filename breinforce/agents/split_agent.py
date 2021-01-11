from breinforce.agents import BaseAgent


class SplitAgent(BaseAgent):
    """
    Agent who can do limited amount of fractual bets.
    """

    def __init__(self, legal_actions, fractions):
        super().__init__(legal_actions)
        self.fractions = fractions

    def act(self, action, split, pot, call, min_raise, max_raise):
        bet = None
        if 'fold' in action:
            bet = 0
        if 'call' in action:
            bet = call
        if 'raise' in action:
            bet = call + split * pot
        if 'allin' in action:
            bet = max_raise
        return bet
