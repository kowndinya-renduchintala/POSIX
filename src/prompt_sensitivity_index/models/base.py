from abc import ABC, abstractmethod
from typing import Tuple

class LanguageModel(ABC):
    @abstractmethod
    def get_responses(self, prompts: list[str], batched=False, **kwargs) -> Tuple[list[list[int]], list[str]]:
        ...

    @abstractmethod
    def _get_response(self, prompt: str, **kwargs) -> Tuple[list[int], str]:
        ...

    @abstractmethod
    def _get_responses_batched(self, prompts: list[str], **kwargs) -> Tuple[list[list[int]], list[str]]:
        ...
    
    @abstractmethod
    def compute_log_probabilties(self, prompts: list[str], responses: list[list[int]], batched=False) -> list[float]:
        ...
    
    @abstractmethod
    def _compute_log_probability(self, prompt: str, response: list[int]) -> float:
        ...
    
    @abstractmethod
    def _compute_log_probabilities_batched(self, prompts: list[str], responses: list[list[int]]) -> list[float]:
        ...