import json

class Config():
    def __init__(self, config_path, format="json"):
        if format == "json":
            self.config = json.load(config_path)
        for name, group in self.config:
            self.__setattr__(name, group)
