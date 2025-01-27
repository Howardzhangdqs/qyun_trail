import toml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config
