from loguru import logger
from opentelemetry import trace

from domains.recommendation.application.port.input import Recommender
from domains.recommendation.application.port.output import Predictor
from domains.recommendation.domain import ItemId

_tracer = trace.get_tracer(__name__)
_RECOMMEND_SPAN_NAME = "recommend"


class RecommenderService(Recommender):
    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    async def recommend(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        log = logger.bind(request_id=request_id)
        with _tracer.start_as_current_span(_RECOMMEND_SPAN_NAME) as span:
            span.set_attribute("request_id", request_id)

            if not user_history:
                log.info("Empty user history; returning no recommendations")
                return []

            return await self._predictor.predict(user_history, request_id=request_id)
