from abc import ABC, abstractmethod


class Flow(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def execute(self, options, args):
        pass
