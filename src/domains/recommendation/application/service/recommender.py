from domains.recommendation.application.port.input import Recommender
from domains.recommendation.application.port.output import Predictor
from domains.recommendation.domain import ItemId


class RecommenderService(Recommender):
    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    async def recommend(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        if not user_history:
            return []
        return await self._predictor.predict(user_history, request_id=request_id)
