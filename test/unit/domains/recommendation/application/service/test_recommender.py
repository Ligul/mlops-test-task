from unittest.mock import AsyncMock, Mock

from domains.recommendation.application.port.output import Predictor
from domains.recommendation.application.service.recommender import RecommenderService
from domains.recommendation.domain import ItemId


async def test_recommend_delegates_to_predictor_for_non_empty_history() -> None:
    # Arrange
    predictor = Mock(spec=Predictor)
    predictor.predict = AsyncMock(return_value=[ItemId(5), ItemId(9)])
    service = RecommenderService(predictor)

    # Act
    result = await service.recommend([ItemId(1), ItemId(2)], request_id="req-1")

    # Assert
    predictor.predict.assert_awaited_once_with([ItemId(1), ItemId(2)], request_id="req-1")
    assert result == [ItemId(5), ItemId(9)]


async def test_recommend_short_circuits_empty_history_without_calling_predictor() -> None:
    # Arrange
    predictor = Mock(spec=Predictor)
    predictor.predict = AsyncMock()
    service = RecommenderService(predictor)

    # Act
    result = await service.recommend([], request_id="req-1")

    # Assert
    assert result == []
    predictor.predict.assert_not_awaited()
