import uuid

import grpc
from loguru import logger

from domains.recommendation.application.port.input import Recommender
from domains.recommendation.domain import ItemId, ItemIdOutOfRangeError
from grpc_proto.recommendation.v1 import recommendation_pb2, recommendation_pb2_grpc

_REQUEST_ID_HEADER = "x-request-id"


def _request_id(context: grpc.aio.ServicerContext) -> str:
    # Use upstream or generate new
    for key, value in context.invocation_metadata() or ():
        if key == _REQUEST_ID_HEADER:
            return value.decode() if isinstance(value, bytes) else value
    return str(uuid.uuid4())


class GrpcRecommender(recommendation_pb2_grpc.RecommenderServiceServicer):
    def __init__(self, recommender: Recommender) -> None:
        self._recommender = recommender

    async def Recommend(  # noqa: N802
        self, request: recommendation_pb2.RecommendRequest, context: grpc.aio.ServicerContext
    ) -> recommendation_pb2.RecommendResponse:
        request_id = _request_id(context)
        log = logger.bind(request_id=request_id)
        log.bind(history_size=len(request.item_ids)).info("Recommend request received")
        try:
            result = await self._recommender.recommend(
                [ItemId(item_id) for item_id in request.item_ids], request_id=request_id
            )
        except ItemIdOutOfRangeError as exc:
            log.bind(item_id=exc.item_id, num_items=exc.num_items).info("Recommend rejected: item id out of range")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        except Exception:
            log.exception("Recommend failed")
            await context.abort(grpc.StatusCode.INTERNAL, "recommendation failed")
            raise
        log.bind(recommendation_count=len(result)).info("Recommend request handled")
        return recommendation_pb2.RecommendResponse(item_ids=[int(item_id) for item_id in result])
