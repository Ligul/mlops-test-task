import uuid
from unittest.mock import AsyncMock, Mock

import grpc
import pytest

from domains.recommendation.adapter.input.grpc_recommender import GrpcRecommender
from domains.recommendation.application.port.input import Recommender
from domains.recommendation.domain import ItemId
from grpc_proto.recommendation.v1 import recommendation_pb2


async def test_recommend_maps_request_ids_to_domain_and_response_back_to_int() -> None:
    # Arrange
    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(return_value=[ItemId(5), ItemId(9)])
    service = GrpcRecommender(recommender)
    request = recommendation_pb2.RecommendRequest(item_ids=[1, 2, 3])
    context = Mock()
    context.invocation_metadata.return_value = (("x-request-id", "req-123"),)

    # Act
    response = await service.Recommend(request, context)

    # Assert
    recommender.recommend.assert_awaited_once_with([ItemId(1), ItemId(2), ItemId(3)], request_id="req-123")
    assert list(response.item_ids) == [5, 9]


async def test_recommend_aborts_with_internal_when_recommendation_fails() -> None:
    # Arrange
    class _AbortError(Exception):
        pass

    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(side_effect=RuntimeError("boom"))
    service = GrpcRecommender(recommender)
    context = AsyncMock()
    context.invocation_metadata = Mock(return_value=())
    # grpc.aio.abort() aborts by raising; its real type is private, so we fake it
    context.abort = AsyncMock(side_effect=_AbortError)
    request = recommendation_pb2.RecommendRequest(item_ids=[1])

    # Act / Assert
    with pytest.raises(_AbortError):
        await service.Recommend(request, context)
    context.abort.assert_awaited_once_with(grpc.StatusCode.INTERNAL, "recommendation failed")


async def test_recommend_propagates_upstream_request_id() -> None:
    # Arrange
    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(return_value=[])
    service = GrpcRecommender(recommender)
    request = recommendation_pb2.RecommendRequest(item_ids=[1])
    context = Mock()
    context.invocation_metadata.return_value = (("x-request-id", "upstream-id"),)

    # Act
    await service.Recommend(request, context)

    # Assert
    _, kwargs = recommender.recommend.call_args
    assert kwargs["request_id"] == "upstream-id"


async def test_recommend_generates_request_id_when_absent() -> None:
    # Arrange
    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(return_value=[])
    service = GrpcRecommender(recommender)
    request = recommendation_pb2.RecommendRequest(item_ids=[1])
    context = Mock()
    context.invocation_metadata.return_value = ()

    # Act
    await service.Recommend(request, context)

    # Assert
    _, kwargs = recommender.recommend.call_args
    assert uuid.UUID(kwargs["request_id"])
