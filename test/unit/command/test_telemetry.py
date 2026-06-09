import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from command.telemetry import build_span_exporter


def test_build_span_exporter_console() -> None:
    assert isinstance(build_span_exporter("console"), ConsoleSpanExporter)


def test_build_span_exporter_otlp() -> None:
    assert isinstance(build_span_exporter("otlp"), OTLPSpanExporter)


def test_build_span_exporter_unknown_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        build_span_exporter("bogus")
