import uuid
from unittest.mock import AsyncMock, Mock

import grpc
import pytest

from domains.recommendation.adapter.input.grpc_recommender import GrpcRecommender
from domains.recommendation.application.port.input import Recommender
from domains.recommendation.domain import ItemId, ItemIdOutOfRangeError
from grpc_proto.recommendation.v1 import recommendation_pb2


class _AbortError(Exception):
    # stand-in for grpc.aio's private abort exception
    pass


@pytest.fixture
def aborting_context() -> AsyncMock:
    context = AsyncMock()
    context.invocation_metadata = Mock(return_value=())
    context.abort = AsyncMock(side_effect=_AbortError)
    return context


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


async def test_recommend_aborts_with_internal_when_recommendation_fails(aborting_context: AsyncMock) -> None:
    # Arrange
    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(side_effect=RuntimeError("boom"))
    service = GrpcRecommender(recommender)
    request = recommendation_pb2.RecommendRequest(item_ids=[1])

    # Act / Assert
    with pytest.raises(_AbortError):
        await service.Recommend(request, aborting_context)
    aborting_context.abort.assert_awaited_once()
    assert aborting_context.abort.await_args.args[0] == grpc.StatusCode.INTERNAL


async def test_recommend_aborts_with_invalid_argument_on_out_of_range_item(aborting_context: AsyncMock) -> None:
    # Arrange
    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(side_effect=ItemIdOutOfRangeError(10000, 10000))
    service = GrpcRecommender(recommender)
    request = recommendation_pb2.RecommendRequest(item_ids=[10000])

    # Act / Assert
    with pytest.raises(_AbortError):
        await service.Recommend(request, aborting_context)
    aborting_context.abort.assert_awaited_once()
    assert aborting_context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


async def test_recommend_aborts_with_internal_on_unexpected_value_error(aborting_context: AsyncMock) -> None:
    # Arrange
    recommender = Mock(spec=Recommender)
    recommender.recommend = AsyncMock(side_effect=ValueError("unexpected"))
    service = GrpcRecommender(recommender)
    request = recommendation_pb2.RecommendRequest(item_ids=[1])

    # Act / Assert
    with pytest.raises(_AbortError):
        await service.Recommend(request, aborting_context)
    aborting_context.abort.assert_awaited_once()
    assert aborting_context.abort.await_args.args[0] == grpc.StatusCode.INTERNAL


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
