from config import load_config, Config

config = load_config('./toml/default.toml')

print(config.__dict__)
