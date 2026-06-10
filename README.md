# Recommendation Service (MLOps Test Task)

**English** | [Русский](README_ru.md)

gRPC service: takes a user's interaction history (a list of `item_id`) and returns a list of recommended `item_id`. Inference runs on a model in ONNX format

The service is deployed and reachable at a public gRPC endpoint - the address and token are attached to the submission message

---------

## 1. Quick Start (local)

Only Docker with Docker Compose is required

```bash
cp .env.template .env   # defaults are enough for a local run
make rebuild            # build the image and start; the service listens on 127.0.0.1:50051
```

How to call the method and check the response - [section 2](#2-verify)

---------

## 2. Verify

The fastest way is the smoke-test client ([`scripts/smoke_test.py`](scripts/smoke_test.py)), it sends a few sample histories, validates the responses and prints the recommendations

Against the live service (address and token from the submission message):

```bash
make smoke-test ADDRESS=<host>:50051 TOKEN=<token>
```

Locally, against the running container (the token is taken from `.env`):

```bash
make smoke-test
```

This is the same script that runs in CI ([`smoke-test.yml`](.github/workflows/smoke-test.yml))

The method can also be called by hand with any gRPC client. The service exposes server reflection and a health check, so `grpcurl` works without proto files:

```bash
grpcurl -plaintext -H "authorization: Bearer <token>" \
  -d '{"item_ids": [1, 67, 100]}' \
  <host>:50051 recommendation.v1.RecommenderService/Recommend
```

Authorization - the `authorization: Bearer <token>` metadata
Full method path - `/recommendation.v1.RecommenderService/Recommend` (on the `RecommenderService` name see [section 3.5](#35-grpc-contract))

---------

## 3. Architecture

### 3.1 Project Layout

Repository map, key directories:

```
.
├── api/proto/recommendation/v1/recommendation.proto  # gRPC contract (Buf)
├── scripts/
│   ├── export_model.py         # PyTorch > ONNX: export + output parity check
│   └── smoke_test.py           # gRPC client (local and in CI)
├── models/model.onnx           # exported model, baked into the image
├── src/
│   ├── command/                # composition root: config, DI, server, logger
│   │   └── grpc_server/        # entrypoint, auth interceptor, DI container
│   ├── domains/recommendation/ # hexagonal core
│   │   ├── domain/             # ItemId, errors (no dependencies)
│   │   ├── application/        # ports (Recommender, Predictor) + service
│   │   └── adapter/            # input/grpc_recommender, output/onnx_predictor
│   ├── telemetry/              # OpenTelemetry tracing setup
│   └── grpc_proto/             # generated stubs (Buf), separate import root
├── test/unit/                  # unit tests across all layers
├── Dockerfile                  # multi-stage, dev/CI/prod via WITH_DEV
├── docker-compose.yml          # local development (binds 127.0.0.1)
├── docker-compose.prod.yml     # VPS (OTLP>Grafana, binds 0.0.0.0)
├── Makefile                    # make targets over docker compose (see make help)
└── .github/workflows/          # ci-cd.yml, smoke-test.yml
```

### 3.2 Hexagonal Layers

Hexagonal architecture, dependencies point inward, toward the core:

- **`domain`** - business types and errors, with no external dependencies ([`ItemId`](src/domains/recommendation/domain/types.py), [`ItemIdOutOfRangeError`](src/domains/recommendation/domain/errors.py))
- **`application`** - the [`Recommender`](src/domains/recommendation/application/port/input/recommender.py) (input) and [`Predictor`](src/domains/recommendation/application/port/output/predictor.py) (output) ports plus the [`RecommenderService`](src/domains/recommendation/application/service/recommender.py) with business logic on top of the ports
- **`adapter`** - port implementations: [`GrpcRecommender`](src/domains/recommendation/adapter/input/grpc_recommender.py) on the input and [`ONNXPredictor`](src/domains/recommendation/adapter/output/onnx_predictor.py) on the output
- **`command`** - composition root: configuration, DI container ([`container.py`](src/command/grpc_server/container.py)), server startup, logger

The core knows nothing about gRPC or ONNX and depends only on the ports. So the transport and the model are swappable without touching business logic, and each layer is tested in isolation: neighboring ports are replaced with mocks

### 3.3 Model Lifecycle

The PyTorch model is exported to ONNX by [`scripts/export_model.py`](scripts/export_model.py) (`make export-model`); at runtime the service works only with the `.onnx`

Key points in the export:

- **Embeddings via `register_buffer`.** In the original class the embedding matrix is a plain attribute (`torch.rand(...)`); such a tensor is not treated as module state and would not make it into the ONNX graph. Registering it as a buffer bakes the weights into the graph
- **Variable history length.** `dynamic_axes` makes the history axis dynamic, so a single model accepts input of any length (opset 17)
- **Vocabulary size is stored in the model.** The number of items is written to the ONNX file metadata (`num_items`) and read by [`ONNXPredictor`](src/domains/recommendation/adapter/output/onnx_predictor.py) on load to validate the `item_id` range. The vocab is not hardcoded in the service: retrain the model with a different vocabulary and the service picks up the new value with no code changes
- **Check against the original.** After the export, `verify()` runs one input through PyTorch and ONNX Runtime and compares the outputs

The resulting [`models/model.onnx`](models/model.onnx) is baked into the Docker image and loaded once at startup as a DI singleton; the path is set via `MODEL_PATH`

### 3.4 Request Flow

The server is `grpc.aio`. The path of a single request:

```
RPC > tracing interceptor > auth interceptor > GrpcRecommender > RecommenderService > ONNXPredictor
```

- **Authorization** ([`auth.py`](src/command/grpc_server/auth.py)): the interceptor checks the bearer token in constant time (`hmac.compare_digest`); health-check methods pass through without a token
- **`request_id`**: [`GrpcRecommender`](src/domains/recommendation/adapter/input/grpc_recommender.py) takes it from `x-request-id` or creates a new one and binds it to the logs and the span, making it the end-to-end correlation key
- **Empty history**: `mean` over an empty tensor would give `nan`, so an empty input is short-circuited to `[]` in the service, without hitting the model
- **Inference does not block the loop**: the CPU-bound `session.run` is offloaded to a thread pool via `asyncio.to_thread`
- **Error codes**: an input `item_id` outside the vocabulary - `INVALID_ARGUMENT`, other failures - `INTERNAL`

### 3.5 gRPC Contract

The contract lives in [`recommendation.proto`](api/proto/recommendation/v1/recommendation.proto). The Python stubs are generated by Buf:

```bash
make proto-gen
```

The generated code is written to `src/grpc_proto`: it is kept separate from the handwritten code and wired in as a separate import root via `PYTHONPATH`

> **Note:** I named the service `RecommenderService`, while the task has `Recommender`: Buf STANDARD lint requires the `Service` suffix. This intentionally differs from the contract in the task, clients call the method as `recommendation.v1.RecommenderService/Recommend`

### 3.6 Configuration

All settings come from environment variables ([`configuration.py`](src/command/configuration.py), pydantic-settings):

- `GRPC_PORT`, `MODEL_PATH`, `ONNX_PROVIDERS`
- the `OTEL_*` family
- `GRPC_TOKEN` - `SecretStr`, required (the service will not start without it)

For a local run the defaults from [`.env.template`](.env.template) are enough

---------

## 4. CI/CD

There are two pipelines in [`.github/workflows`](.github/workflows):

**[`ci-cd.yml`](.github/workflows/ci-cd.yml)** - the main one, on `master` events:

- **PR to `master`**: lint (`black`, `ruff`, `pyright`) and unit tests
- **push to `master`**: the same + building and publishing the prod image to GHCR (tags: short SHA and `latest`), then a deploy to the VPS - copying [`docker-compose.prod.yml`](docker-compose.prod.yml) and `pull` + `up` over SSH
\
  Lint and tests run in the same Docker image ([`Dockerfile`](Dockerfile)) as prod, but with `WITH_DEV=1` (dev dependencies on top of a shared base), so what is checked is effectively the same build that ships to prod

**[`smoke-test.yml`](.github/workflows/smoke-test.yml)** - manual (`workflow_dispatch`): pulls `latest` from GHCR and runs the smoke client against the public endpoint

The deploy and the smoke test require a VPS that is already up and the secrets in place - see [section 5](#5-vps-setup-and-secrets)

---------

## 5. VPS Setup and Secrets

**1. Run once on the server:**

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

sudo mkdir -p /opt/recommender && sudo chown $USER:$USER /opt/recommender

sudo ufw allow 50051/tcp
```

**2. Add secrets** (GitHub: Settings > Secrets and variables > Actions):

| Secret | Purpose |
|---|---|
| `GHCR_TOKEN` | PAT with `read:packages` scope (the VPS pulls images from GHCR) |
| `SSH_HOST` | server IP or domain |
| `SSH_USER` | SSH user |
| `SSH_PRIVATE_KEY` | private SSH key for the deploy |
| `GRPC_TOKEN` | bearer token the service checks |
| `OTEL_EXPORTER_OTLP_HEADERS` | Grafana Cloud OTLP authorization header (keep `%20` as is) |

`GRPC_TOKEN` is the single source of truth: the pipeline injects it into the container at deploy time and reuses it in the smoke workflow; it is not stored on the server

There are two GHCR tokens: the built-in `GITHUB_TOKEN` pushes the image during the build and expires with the run, while `GHCR_TOKEN` (PAT) is needed for the VPS to pull the image later, for example on a restart

---------

## 6. Observability (Logging and Tracing)

The logs are structured (loguru, [`logger.py`](src/command/logger.py)): JSON in prod, human-readable output locally. Each request gets a `request_id` (from `x-request-id` or generated), bound to all of its log lines; the input (`history_size`), the outcome (`recommendation_count`) and rejections/errors are logged

Tracing is OpenTelemetry ([`tracing.py`](src/telemetry/tracing.py)). Each `Recommend` RPC produces a server span, its duration and gRPC status are captured automatically by the gRPC interceptor. Inside it is a child span `recommend` from the application service: it separates the timing of the recommendations themselves from the gRPC overhead and carries the same `request_id` as the logs. This is the key that correlates logs and traces

> **Note on architecture:** the `recommend` span lives in the application service. A stricter hexagonal reading would keep OpenTelemetry out of the core as an external dependency. In practice a tracer is an ambient cross-cutting concern, like the logger, so instrumenting the service is a deliberate, pragmatic trade-off

Trace export is set by the environment, the application code is identical across environments:

- local: `OTEL_TRACES_EXPORTER=console` - spans to stdout (`make logs`)
- prod: `otlp` - straight to Grafana Cloud over OTLP/HTTP

![Recommend RPC trace in Grafana Cloud](docs/images/grafana-trace.png)

---------

## 7. Testing

**Unit tests** cover all layers - the auth interceptor, the gRPC adapter, the ONNX predictor, the application service and the telemetry; they run in the same Docker image as the build, including in CI ([section 4](#4-cicd)):

```bash
make test-unit
```

**The smoke test** ([`scripts/smoke_test.py`](scripts/smoke_test.py)) is an end-to-end check of the live service: it sends a set of histories (short, long, single-item and empty) and validates the responses. The commands for local, remote and CI runs are in [section 2](#2-verify)

---------

## 8. Development

The whole cycle goes through Docker and Make ([`Makefile`](Makefile)), `make help` lists all targets:

- container: `rebuild`, `shell`, `logs`, `down`
- code quality: `format`, `lint`, `test` (`black`, `ruff`, `pyright`, `pytest`)
- model and proto: `export-model`, `proto-lint`, `proto-gen`

In dev mode the sources are mounted into the container, so `format`/`lint`/`test` work with the current code without rebuilding the image. A local `poetry install --with dev` is only needed for IDE highlighting and type checks - it is not required to run the service

---------

## 9. Possible Improvements and Scaling

The service deliberately stays within the scope of the task. A few directions for further work:

- **Model registry and versioning.** Right now the `.onnx` is baked into the image, so the model version is tied to the git SHA. The logical step is to move the artifact into a registry (MLflow Model Registry or S3 with a version tag) and load it by `MODEL_VERSION`, decoupling the model release from the code release. The `num_items` baked into the file is already a step toward a self-describing artifact
- **A/B testing through the port.** `Predictor` is a port, so A/B fits in as a router adapter that splits traffic between two model versions by a hash of `request_id`. The chosen variant is set as a span attribute and a log field - the same mechanics as `request_id`, so offline attribution of the results is almost free
- **Canary through the orchestrator.** A gradual rollout of a new version at the Kubernetes level (a fraction of replicas on the new image), independent of the service code
- **Horizontal scaling.** The service is stateless (the model is read-only, loaded into each process), so it scales out with N replicas behind a gRPC-aware load balancer. The health check and reflection are already in place for readiness/liveness probes
- **Catalog scale.** Right now recommendations are a full `matmul + topk`, that is O(N) over the catalog; on large catalogs an ANN index (FAISS, ScaNN) would replace it. The ONNX providers are exposed via `ONNX_PROVIDERS`, so switching to a GPU or a quantized model is a matter of configuration
- **Metrics.** There are logs and traces now, but no metrics - Prometheus is the natural addition (RPS, latency, error rate) with alerts on top
