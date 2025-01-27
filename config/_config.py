from typing import Dict
import timm
import toml


model_size = [0.5, 1, 1.5, 2]


class Config:
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}

        self.dataset_path = config.get('dataset_path', './dataset')
        self.model_backbone = config["model"].get('backbone', 'resnet18')

        # 判断模型是否在timm模型库中
        if self.model_backbone not in timm.list_models():
            raise ValueError(f"Model {self.model_backbone} not in timm model list")

        self.model_size = config["model"].get('size', 0.5)

        if self.model_size not in model_size:
            raise ValueError(f"Model size {self.model_size} not in {model_size}")

        self.model_output_num = config.get('model_output_num', 1)


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config = Config(toml.load(f))
    return config
