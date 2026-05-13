from abc import ABC, abstractmethod
from PIL import Image


class VLMBase(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str:
        ...
