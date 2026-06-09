from unittest.mock import Mock

import numpy as np
import pytest
from loguru import logger

from domains.recommendation.adapter.output.onnx_predictor import ONNXPredictor
from domains.recommendation.domain import ItemId, ItemIdOutOfRangeError


def _make_predictor(
    *,
    run_return=None,
    run_side_effect=None,
    num_items: int = 10000,
) -> tuple[ONNXPredictor, Mock]:
    # Bypass __init__ so no real ONNX model/session is loaded
    predictor = ONNXPredictor.__new__(ONNXPredictor)
    session = Mock()
    if run_side_effect is not None:
        session.run.side_effect = run_side_effect
    else:
        session.run.return_value = run_return
    predictor._session = session
    predictor._input_name = "user_history"
    predictor._output_name = "recommendations"
    predictor._num_items = num_items
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


@pytest.mark.parametrize(
    "bad_id",
    [-1, 10000, 99999],
    ids=["negative", "equals-num-items", "above-range"],
)
async def test_predict_rejects_item_id_out_of_range(bad_id: int) -> None:
    # Arrange
    predictor, session = _make_predictor(num_items=10000)

    # Act / Assert
    with pytest.raises(ItemIdOutOfRangeError):
        await predictor.predict([ItemId(3), ItemId(bad_id)], request_id="req-1")

    session.run.assert_not_called()


async def test_predict_accepts_boundary_item_ids() -> None:
    # Arrange
    predictor, session = _make_predictor(run_return=[np.array([1], dtype=np.int64)], num_items=10000)

    # Act
    await predictor.predict([ItemId(0), ItemId(9999)], request_id="req-1")

    # Assert
    session.run.assert_called_once()


def test_read_num_items_parses_metadata_value() -> None:
    # Arrange
    predictor, session = _make_predictor()
    session.get_modelmeta.return_value.custom_metadata_map = {"num_items": "10000"}

    # Act / Assert
    assert predictor._read_num_items() == 10_000


def test_read_num_items_raises_when_metadata_missing() -> None:
    # Arrange
    predictor, session = _make_predictor()
    session.get_modelmeta.return_value.custom_metadata_map = {}

    # Act / Assert
    with pytest.raises(ValueError, match="num_items"):
        predictor._read_num_items()
