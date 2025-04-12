import os
import yaml


class Config(dict):
    def __init__(self, config_path):
        super(Config, self).__init__()
        with open(config_path, 'r', encoding="utf-8") as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        return None

    def update_key(self, name, value):
        if self._dict.get(name) is not None:
            self._dict[name] = value
        else:
            raise KeyError(f"Key {name} not found in config")


def load_config(config_path):
    config = Config(config_path)
    return config
