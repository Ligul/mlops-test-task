from loguru import logger

from domains.recommendation.application.port.input import Recommender
from domains.recommendation.application.port.output import Predictor
from domains.recommendation.domain import ItemId, ItemIdOutOfRangeError
from telemetry.tracing import traced_operation

_RECOMMEND_SPAN_NAME = "recommend"


class RecommenderService(Recommender):
    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    async def recommend(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        log = logger.bind(request_id=request_id)
        with traced_operation(
            _RECOMMEND_SPAN_NAME, request_id=request_id, non_fault_exceptions=(ItemIdOutOfRangeError,)
        ):
            if not user_history:
                log.info("Empty user history; returning no recommendations")
                return []

            return await self._predictor.predict(user_history, request_id=request_id)
