from utils import load_config, Config

config = load_config('./cfg/default.toml')

if __name__ == "__main__":
    from rich import print
    print(config.__dict__)
