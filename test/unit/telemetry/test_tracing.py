from unittest.mock import MagicMock

import pytest
from opentelemetry.trace import StatusCode

from telemetry import traced_operation


@pytest.fixture
def span(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    tracer = MagicMock()
    cm = tracer.start_as_current_span.return_value
    cm.__exit__.return_value = False  # let exceptions propagate out of the span
    monkeypatch.setattr("telemetry.tracing._tracer", tracer)
    return cm.__enter__.return_value


def test_sets_request_id_attribute(span: MagicMock) -> None:
    # Act
    with traced_operation("op", request_id="req-1"):
        pass

    # Assert
    span.set_attribute.assert_called_once_with("request_id", "req-1")


def test_marks_span_error_on_fault(span: MagicMock) -> None:
    # Act
    with pytest.raises(RuntimeError), traced_operation("op", request_id="req-1"):
        raise RuntimeError()

    # Assert
    span.record_exception.assert_called_once()
    span.set_status.assert_called_once()
    (status,), _ = span.set_status.call_args
    assert status.status_code == StatusCode.ERROR


def test_does_not_mark_span_error_for_non_fault(span: MagicMock) -> None:
    # Arrange
    class _NonFaultError(Exception):
        pass

    # Act
    with (
        pytest.raises(_NonFaultError),
        traced_operation("op", request_id="req-1", non_fault_exceptions=(_NonFaultError,)),
    ):
        raise _NonFaultError

    # Assert
    span.record_exception.assert_not_called()
    span.set_status.assert_not_called()
