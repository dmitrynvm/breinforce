from breinforce import configs, envs


def configure():
    """
    Merges the local configured envs to the global OpenAI Gym list.
    """
    try:
        env_configs = {}
        for name, config in configs.bropoker.__dict__.items():
            if not name.endswith('_PLAYER'):
                continue
            keywords = [sub_string.title() for sub_string in name.split('_')]
            env_id = ''.join(keywords)
            env_id += '-v0'
            env_configs[env_id] = config
        envs.register(env_configs)
    except ImportError:
        pass
