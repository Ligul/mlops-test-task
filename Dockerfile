FROM python:3.11-slim AS builder

ARG WITH_DEV=0

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --no-cache-dir poetry==1.8.5

ARG GRPC_HEALTH_PROBE_VERSION=v0.4.24
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates wget; \
    rm -rf /var/lib/apt/lists/*; \
    wget -qO /bin/grpc_health_probe "https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64"; \
    chmod +x /bin/grpc_health_probe

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

RUN poetry config virtualenvs.create false && \
    poetry install $(test "$WITH_DEV" = "1" && echo "--with=dev") --no-root --no-interaction --no-ansi

FROM python:3.11-slim

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app/src/grpc_proto

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /bin/grpc_health_probe /bin/grpc_health_probe

RUN useradd --create-home --shell /bin/false appuser

COPY --chown=appuser:appuser . /app
USER appuser

EXPOSE 50051

CMD ["env"]