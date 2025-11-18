from abc import ABC, abstractmethod
from pydantic import BaseModel


class LLMBase(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        pass