"""
Smoke-test client for the Recommender gRPC service

Usage:
    python scripts/smoke_test.py [--address HOST:PORT] [--token TOKEN] [--timeout SECONDS]

Sends sample user histories to the Recommend RPC and validates the responses
Exits non-zero on any failure
"""

import argparse
import logging
import os
import sys

import grpc

from grpc_proto.recommendation.v1 import recommendation_pb2, recommendation_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


NUM_ITEMS = 10_000

DEFAULT_ADDRESS = "recommender:50051"
DEFAULT_TIMEOUT = 5.0

SAMPLE_HISTORIES: list[tuple[str, list[int]]] = [
    ("typical", [1, 67, 100, 7, 2048]),
    ("single-item", [67]),
    ("long", list(range(32))),
    ("empty", []),
]


def validate_recommendations(history: list[int], recommendations: list[int]) -> None:
    if not history:
        if recommendations:
            msg = f"empty history must yield no recommendations, got {recommendations}"
            raise ValueError(msg)
        return

    if not recommendations:
        msg = "non-empty history must yield at least one recommendation, got none"
        raise ValueError(msg)

    out_of_range = [r for r in recommendations if not 0 <= r < NUM_ITEMS]
    if out_of_range:
        msg = f"recommendations out of range [0, {NUM_ITEMS}): {out_of_range}"
        raise ValueError(msg)


def call_recommend(
    stub: recommendation_pb2_grpc.RecommenderServiceStub,
    history: list[int],
    token: str,
    timeout: float,
) -> list[int]:
    request = recommendation_pb2.RecommendRequest(item_ids=history)
    metadata = [("authorization", f"Bearer {token}")]
    response = stub.Recommend(request, metadata=metadata, timeout=timeout)
    return list(response.item_ids)


def run(address: str, token: str, timeout: float) -> bool:
    log.info("connecting to %s", address)
    failures = 0
    with grpc.insecure_channel(address) as channel:
        stub = recommendation_pb2_grpc.RecommenderServiceStub(channel)
        for name, history in SAMPLE_HISTORIES:
            try:
                recommendations = call_recommend(stub, history, token, timeout)
            except grpc.RpcError as e:
                log.error("[%s] RPC failed: %s %s", name, e.code(), e.details())
                failures += 1
                continue
            try:
                validate_recommendations(history, recommendations)
            except ValueError as e:
                log.error("[%s] invalid response: %s", name, e)
                failures += 1
                continue
            log.info("[%s] history=%s -> recommendations=%s", name, history, recommendations)
    return failures == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--token", default=os.environ.get("GRPC_TOKEN"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        log.error("no auth token: pass --token or set GRPC_TOKEN")
        sys.exit(1)
    if run(args.address, args.token, args.timeout):
        log.info("smoke test PASSED")
        return
    log.error("smoke test FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
