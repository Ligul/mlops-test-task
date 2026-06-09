import asyncio
import signal

import grpc
import grpc.aio
from dependency_injector.wiring import Provide, inject
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from loguru import logger

from command.grpc_server.auth import StaticTokenValidationInterceptor
from command.grpc_server.container import Config, Container
from domains.recommendation.adapter.input.grpc_recommender import GrpcRecommender
from grpc_proto.recommendation.v1 import recommendation_pb2
from grpc_proto.recommendation.v1.recommendation_pb2_grpc import add_RecommenderServiceServicer_to_server
from telemetry import setup_tracing, tracing_interceptors


@inject
async def serve(
    grpc_recommender: GrpcRecommender = Provide[Container.grpc_recommender],
    config: Config = Provide[Container.config_obj],
) -> None:
    logger.info("GRPC server starting...")
    tracer_provider = setup_tracing(config.OTEL_TRACES_EXPORTER, config.OTEL_SERVICE_NAME, config.ENV)
    server = grpc.aio.server(
        interceptors=[
            *tracing_interceptors(),
            StaticTokenValidationInterceptor(config.GRPC_TOKEN.get_secret_value()),
        ],
    )
    add_RecommenderServiceServicer_to_server(grpc_recommender, server)
    health_servicer = health.aio.HealthServicer()  # pyright: ignore[reportAttributeAccessIssue]
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    recommender_service_name = recommendation_pb2.DESCRIPTOR.services_by_name["RecommenderService"].full_name
    health_service_name = health_pb2.DESCRIPTOR.services_by_name["Health"].full_name
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set(recommender_service_name, health_pb2.HealthCheckResponse.SERVING)
    service_names = (
        recommender_service_name,
        health_service_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    server.add_insecure_port(f"[::]:{config.GRPC_PORT}")
    await server.start()
    logger.info(f"GRPC server started on port {config.GRPC_PORT}!")

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)
    await shutdown.wait()

    logger.info("GRPC server stopping...")
    await server.stop(grace=5.0)
    tracer_provider.shutdown()
    logger.info("GRPC server stopped")


if __name__ == "__main__":
    container = Container()
    container.wire(modules=[__name__])
    asyncio.run(serve())
