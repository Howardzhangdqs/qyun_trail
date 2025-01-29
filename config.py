from utils import load_config, Config

config = load_config('./config/default.toml')

print(config.__dict__)
