from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

_tracer = trace.get_tracer(__name__)


@contextmanager
def traced_operation(
    name: str,
    *,
    request_id: str,
    non_fault_exceptions: tuple[type[Exception], ...] = (),
) -> Iterator[Span]:
    with _tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        span.set_attribute("request_id", request_id)
        try:
            yield span
        except non_fault_exceptions:
            raise
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
