from abc import ABC, abstractmethod

from domains.recommendation.domain import ItemId


class Recommender(ABC):
    @abstractmethod
    async def recommend(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        raise NotImplementedError
