from collections import namedtuple

Action = namedtuple('Action', ['name', 'value'])

Episode = namedtuple('Episode', ['state', 'player', 'action'])
