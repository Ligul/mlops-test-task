from abc import ABC, abstractmethod

from domains.recommendation.domain import ItemId


class Predictor(ABC):
    @abstractmethod
    async def predict(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        raise NotImplementedError
