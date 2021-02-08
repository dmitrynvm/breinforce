import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

ENVS_DIR = os.path.join(BASE_DIR, 'envs')

VIEW_DIR = os.path.join(BASE_DIR, 'templates')
