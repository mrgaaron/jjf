from abc import ABC, abstractmethod


class MusicSystem(ABC):
    @abstractmethod
    def pitches(self):
        ...
