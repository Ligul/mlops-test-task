import asyncio
from typing import cast

import numpy as np
import onnxruntime as ort
from loguru import logger

from domains.recommendation.application.port.output import Predictor
from domains.recommendation.domain import ItemId, ItemIdOutOfRangeError

_NUM_ITEMS_METADATA_KEY = "num_items"


class ONNXPredictor(Predictor):
    def __init__(self, model_path: str, providers: list[str]) -> None:
        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._num_items = self._read_num_items()
        except Exception:
            logger.bind(model_path=model_path).exception("Failed to load ONNX model")
            raise
        logger.bind(model_path=model_path, providers=providers, num_items=self._num_items).info("ONNX model loaded")

    async def predict(self, user_history: list[ItemId], request_id: str) -> list[ItemId]:
        if not user_history:
            logger.bind(request_id=request_id).warning(
                "predict called with empty user_history; returning no recommendations"
            )
            return []

        self._validate(user_history)

        input_data = np.array(user_history, dtype=np.int64)

        outputs = await asyncio.to_thread(self._session.run, [self._output_name], {self._input_name: input_data})

        recommendations = cast("np.ndarray", outputs[0])
        return [ItemId(int(idx)) for idx in recommendations]

    def _read_num_items(self) -> int:
        raw = self._session.get_modelmeta().custom_metadata_map.get(_NUM_ITEMS_METADATA_KEY)
        if raw is None:
            message = f"model is missing required metadata {_NUM_ITEMS_METADATA_KEY!r}; re-export the model"
            raise ValueError(message)
        return int(raw)

    def _validate(self, user_history: list[ItemId]) -> None:
        for item_id in user_history:
            if item_id < 0 or item_id >= self._num_items:
                raise ItemIdOutOfRangeError(item_id, self._num_items)
