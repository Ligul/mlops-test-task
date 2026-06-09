import grpc
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import aio_server_interceptor, filters
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

from command.configuration import Config

_CONSOLE = "console"
_OTLP = "otlp"


def build_span_exporter(kind: str) -> SpanExporter:
    if kind == _CONSOLE:
        return ConsoleSpanExporter()
    if kind == _OTLP:
        return OTLPSpanExporter()
    msg = f"Unsupported OTEL_TRACES_EXPORTER: {kind}"
    raise ValueError(msg)


def setup_tracing(config: Config) -> TracerProvider:
    resource = Resource.create({"service.name": config.OTEL_SERVICE_NAME, "deployment.environment": config.ENV})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(build_span_exporter(config.OTEL_TRACES_EXPORTER)))
    trace.set_tracer_provider(provider)
    return provider


def tracing_interceptors() -> list[grpc.aio.ServerInterceptor]:
    return [aio_server_interceptor(filter_=filters.negate(filters.health_check()))]
