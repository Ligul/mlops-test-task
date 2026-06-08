#!/usr/bin/make

PROJECT_DIR ?= $(shell pwd)
BUF_IMG ?= bufbuild/buf:1.70.0

#
# Help
#

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

#
# Local Docker
#

.PHONY: rebuild
rebuild: ## Build and start all containers in detached mode
	docker compose up --build -d

.PHONY: shell
shell: ## Open a bash shell inside the running recommender container
	docker compose exec recommender bash

.PHONY: logs
logs: ## Follow container logs
	docker compose logs -f

.PHONY: down
down: ## Stop and remove containers
	docker compose down --remove-orphans

#
# Code quality
#

.PHONY: format
format: ## Auto-format code with black and ruff
	@docker compose run --rm recommender bash -c "black src/ test/ && ruff check --fix-only --unsafe-fixes src/ test/"

.PHONY: lint
lint: ## Check code style and types (black, ruff, pyright)
	@docker compose run --rm recommender bash -c "black --check src/ && ruff check src/ && pyright"

.PHONY: test
test: ## Run all tests
	@docker compose run --rm recommender bash -c "pytest"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@docker compose run --rm recommender bash -c "pytest test/unit"

#
# Model
#

MODEL_OUTPUT ?= models/model.onnx
MODEL_SEED ?= 67

.PHONY: export-model
export-model: ## Export PyTorch model to ONNX and verify (OUTPUT=... SEED=...)
	@docker compose run --rm recommender python scripts/export_model.py --output $(MODEL_OUTPUT) --seed $(MODEL_SEED)

#
# gRPC
#

BUF_RUN ?= docker run --rm --platform linux/amd64 --volume "$(PROJECT_DIR):/workspace" --workdir /workspace $(BUF_IMG)

.PHONY: proto-lint
proto-lint: ## Lint proto files with buf
	$(BUF_RUN) lint

.PHONY: proto-gen
proto-gen: proto-lint ## Generate gRPC stubs from proto files
	$(BUF_RUN) generate
