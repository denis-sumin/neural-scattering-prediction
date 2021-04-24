from abc import ABC, abstractmethod
from typing import Optional


class DataInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()


from .data import Data  # noqa: E402
from .data_3d import Data3D  # noqa: E402
from .data_planar import DataPlanar  # noqa: E402
