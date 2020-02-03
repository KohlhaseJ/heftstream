from abc import ABCMeta, abstractmethod

class BaseSelector(metaclass=ABCMeta):
    @abstractmethod
    def select(self, X, y):
        pass