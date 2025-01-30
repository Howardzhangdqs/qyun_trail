import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def norm_join(*args):
    """规范化路径拼接"""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), *args))


class Config:
    """统一配置类，包含所有配置项的默认值和嵌套处理"""

    def __init__(self, data: dict = None, base_dir: str = os.path.dirname(__file__)):
        # 顶层配置默认值
        self.dataset_path = norm_join(base_dir, "../dataset")
        self.preprocessed_path = norm_join(base_dir, "../preprocessed")
        self.img_size = [320, 240]
        self.dataset_propotion = 0.8

        # 嵌套配置默认值
        self.model = self.ModelConfig()

        self.dataset_path_train = norm_join(self.dataset_path, "train")
        self.dataset_path_test = norm_join(self.dataset_path, "test")

        # 如果传入数据则更新配置
        if data:
            self._update(data)

    def _update(self, data: dict):
        """递归更新配置数据，保留未指定的默认值"""
        for key, value in data.items():
            # 跳过未定义的配置项
            if not hasattr(self, key):
                continue

            current_value = getattr(self, key)

            # 处理嵌套配置
            if isinstance(current_value, Config) and isinstance(value, dict):
                current_value._update(value)
            elif isinstance(current_value, list) and isinstance(value, list):
                setattr(self, key, value)
            else:
                setattr(self, key, value)

        self._hook()

    def _hook(self):
        self.dataset_path = norm_join(self.dataset_path)
        self.preprocessed_path = norm_join(self.preprocessed_path)
        self.dataset_path_train = norm_join(self.dataset_path, "train")
        self.dataset_path_test = norm_join(self.dataset_path, "test")

    class ModelConfig:
        """内置模型配置类"""

        def __init__(self):
            self.backbone = "resnet18"
            self.size = 0.5
            self.output_num = 1


def load_config(config_path: str) -> Config:
    """加载TOML配置文件"""
    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)
    return Config(toml_data, base_dir=os.path.dirname(config_path))


# 使用示例
if __name__ == "__main__":
    # 测试默认值
    default_config = Config()
    print("Default dataset path:", default_config.dataset_path)
    print("Default model backbone:", default_config.model.backbone)

    # 测试加载配置文件
    config = load_config("config.toml")
    print("\nLoaded dataset path:", config.dataset_path)
    print("Loaded model size:", config.model.size)

    # 测试保留未配置的默认值
    print("Model output_num (default):", config.model.output_num)
