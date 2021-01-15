class BaseView:
    """Base class for renderer. Any renderer must subclass this renderer
    and implement the function render

    Parameters
    ----------
    n_players : int
        number of player
    n_hole_cards : int
        number of hole cards
    n_community_cards : int
        number of community cards
    """

    def __init__(self) -> None:
        pass

    def render(self) -> None:
        """Render the table based on the table configuration

        Parameters
        ----------
        state : dict
            state configuration dictionary,
                state = {
                    "player": int - position of active player,
                    "active": List[bool] - list of active players,
                    "allin": List[bool] - list of all in players,
                    "community_cards": List[Card] - list of community
                                       cards,
                    "dealer": int - position of dealer,
                    "done": bool - list of done players,
                    "hole_cards": List[List[Card]] - list of hole cards,
                    "pot": int - chips in pot,
                    "payouts": List[int] - list of chips won for each
                               player,
                    "prev_action": Tuple[int, int, int] - last
                                   position bet and fold,
                    "street_commits": List[int] - list of number of
                                      chips added to pot from each
                                      player on current street,
                    "stacks": List[int] - list of stack sizes,
                }
        """
        raise NotImplementedError
