from unittest.mock import Mock

import numpy as np
import pytest
from loguru import logger

from domains.recommendation.adapter.output.onnx_predictor import ONNXPredictor
from domains.recommendation.domain import ItemId


def _make_predictor(*, run_return=None, run_side_effect=None) -> tuple[ONNXPredictor, Mock]:
    # Bypass __init__ so no real ONNX model/session is loaded
    predictor = ONNXPredictor.__new__(ONNXPredictor)
    session = Mock()
    if run_side_effect is not None:
        session.run.side_effect = run_side_effect
    else:
        session.run.return_value = run_return
    predictor.session = session
    predictor.input_name = "user_history"
    predictor.output_name = "recommendations"
    return predictor, session


async def test_predict_translates_history_to_int64_and_output_to_plain_int_ids() -> None:
    # Arrange
    predictor, session = _make_predictor(run_return=[np.array([5, 9, 1], dtype=np.int64)])

    # Act
    result = await predictor.predict([ItemId(3), ItemId(7)], request_id="req-1")

    # Assert
    # Output boundary: numpy.int64 must be unwrapped to plain ints
    assert result == [ItemId(5), ItemId(9), ItemId(1)]
    assert all(type(item) is int for item in result)
    # Input boundary: ids forwarded to the session as an int64 array
    (output_names, feed), _ = session.run.call_args
    assert output_names == ["recommendations"]
    assert feed["user_history"].dtype == np.int64
    np.testing.assert_array_equal(feed["user_history"], np.array([3, 7], dtype=np.int64))


async def test_predict_logs_and_returns_empty_for_empty_history() -> None:
    # Arrange
    predictor, session = _make_predictor(run_return=[np.array([1], dtype=np.int64)])
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")

    # Act
    try:
        result = await predictor.predict([], request_id="req-1")
    finally:
        logger.remove(sink_id)

    # Assert
    assert result == []
    session.run.assert_not_called()
    assert [message.record["level"].name for message in messages] == ["WARNING"]


async def test_predict_propagates_inference_error() -> None:
    # Arrange
    predictor, _ = _make_predictor(run_side_effect=RuntimeError("boom"))

    # Act / Assert
    with pytest.raises(RuntimeError, match="boom"):
        await predictor.predict([ItemId(3)], request_id="req-1")
