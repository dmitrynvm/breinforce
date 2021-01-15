
def parse_config(config: dict) -> dict:
    for key, vals in config.items():
        if type(vals) == str:
            if vals == "inf":
                config[key] = float("inf")
        if type(vals) == list:
            for i, val in enumerate(vals):
                if type(val) == str:
                    if val == "inf":
                        vals[i] = float("inf")
    return config
