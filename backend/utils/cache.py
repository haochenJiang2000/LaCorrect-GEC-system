import importlib
import pickle

class Cache(object):
    """
    处理缓存
    在 uWSGI 中运行的时候，使用 UWSGI 的缓存机制，实现进程间共享
    否则，缓存到一个 dict 中
    """
    def __init__(self):
        self.__g = None
        try:
            self.__uwsgi = importlib.import_module('uwsgi')
            print('USE CACHE UWSGI')
        except:
            self.__g = {}
            print('USE CACHE MEMORY')

    def _getuwsgicache(self, name):
        """
        获取 UWSGI 缓存
        :param name: 真实的 name，带有 regional 信息
        :return: 序列化之后的 python 对象
        """
        raw_value = self.__uwsgi.cache_get(name)
        # print('_getuwsgicache:', raw_value)
        if raw_value is not None:
            return pickle.loads(raw_value, encoding='utf8')
        return None

    def _setuwsgicache(self, name, value):
        """
        设置 UWSGI 缓存
        :param name: 设置名称
        :param value: 值
        :return:
        """
        if value is None:
            self.__uwsgi.cache_del(name)
            return
        raw_value = pickle.dumps(value)
        # print('_setuwsgicache:', raw_value)
        if self.__uwsgi.cache_exists(name):
            self.__uwsgi.cache_update(name, raw_value)
        else:
            self.__uwsgi.cache_set(name, raw_value)

    def get(self, name):
        if self.__g is None:
            return self._getuwsgicache(name)
        return self.__g.get(name, None)

    def set(self, name, value):
        if self.__g is None:
            self._setuwsgicache(name, value)
        else:
            self.__g[name] = value