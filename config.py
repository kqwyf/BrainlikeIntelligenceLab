"""
使用get_namespace，将json格式的配置文件转为命名空间。
比如可以用`ns.a.b.c`获得`{"a": {"b": {"c": "key"}}}`中的"key"
"""
import json


class NameSpace(dict):

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, data):
        if isinstance(data, str):
            data = json.loads(data)

        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getattr__(self, attr):
        return self.get(attr, None)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set)):
            return [self._wrap(v) for v in value]
        if isinstance(value, dict):
            return NameSpace(value)
        return value


def get_namespace(path: str):
    with open(path, encoding="utf8") as f:
        _cfg = json.load(f)
    return NameSpace(_cfg)


if __name__ == "__main__":
    ns = get_namespace("./config.json")
    print(ns.train)
    print(ns.data.train)
    print(ns.data.train.path)
    print(ns.model)
