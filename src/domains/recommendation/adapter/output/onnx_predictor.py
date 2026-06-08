import asyncio
from typing import cast

import numpy as np
import onnxruntime as ort
from loguru import logger

from domains.recommendation.application.port.output import Predictor
from domains.recommendation.domain import ItemId


class ONNXPredictor(Predictor):
    def __init__(self, model_path: str, providers: list[str]) -> None:
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception:
            logger.bind(model_path=model_path).exception("Failed to load ONNX model")
            raise
        logger.bind(model_path=model_path, providers=providers).info("ONNX model loaded")

    async def predict(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        if not user_history:
            logger.bind(request_id=request_id).warning(
                "predict called with empty user_history; returning no recommendations"
            )
            return []

        input_data = np.array(user_history, dtype=np.int64)

        outputs = await asyncio.to_thread(self.session.run, [self.output_name], {self.input_name: input_data})

        recommendations = cast("np.ndarray", outputs[0])
        return [ItemId(int(idx)) for idx in recommendations]
