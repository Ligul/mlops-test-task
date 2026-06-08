from dependency_injector import containers, providers

from command.configuration import Config
from command.logger import config_logger
from domains.recommendation.adapter.input.grpc_recommender import GrpcRecommender
from domains.recommendation.adapter.output.onnx_predictor import ONNXPredictor
from domains.recommendation.application.service.recommender import RecommenderService


class Container(containers.DeclarativeContainer):
    config = Config()

    config_logger(config.LOGGING_LEVEL, config.ENV)

    config_obj = providers.Object(config)

    predictor = providers.Singleton(
        ONNXPredictor,
        model_path=config.MODEL_PATH,
        providers=config.ONNX_PROVIDERS,
    )

    recommender_service = providers.Singleton(
        RecommenderService,
        predictor=predictor,
    )

    grpc_recommender = providers.Singleton(
        GrpcRecommender,
        recommender=recommender_service,
    )
