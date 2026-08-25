.DEFAULT_GOAL := help
.PHONY: help infra up down logs reset seed eval bench fmt lint test psql mcp \
        venv install api worker web dev stop-app

# Two run modes, and the split matters for how it feels to work on this.
#
#   make dev   infra in Docker, app native. Nothing to rebuild, sub-second
#              reloads, breakpoints work. This is the day-to-day loop.
#   make up    everything in Docker. One command from a cold clone.
#
# Even `make up` mounts the source and reloads on save — a code change never
# needs a rebuild, only a dependency change does.

VENV    := .venv
PY      := $(VENV)/Scripts/python.exe
ifeq ($(OS),)
PY      := $(VENV)/bin/python
endif

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- infra

infra: ## Start infra only (postgres, qdrant, valkey, rabbitmq, minio)
	docker compose up -d --wait
	@echo ""
	@echo "  postgres   localhost:5432   (ragent / ragent)"
	@echo "  qdrant     http://localhost:6333/dashboard"
	@echo "  rabbitmq   http://localhost:15672   (ragent / ragent)"
	@echo "  minio      http://localhost:9001    (ragent / ragentragent)"

dev: infra ## Start infra, then print how to run the app natively
	@echo ""
	@echo "  Infra is up. Run the app in three terminals:"
	@echo "    make api        reloads on save"
	@echo "    make worker     reloads on save"
	@echo "    make web        http://localhost:3000"
	@echo ""
	@echo "  Office formats also need LibreOffice, which stays in Docker:"
	@echo "    make worker-convert"
	@echo ""
	@echo "  Then: make seed"

up: ## Start everything in Docker, app included
	docker compose --profile app up -d --wait --build
	@echo ""
	@echo "  web        http://localhost:3000"
	@echo "  api docs   http://localhost:8000/docs"
	@echo "  rabbitmq   http://localhost:15672   (ragent / ragent)"
	@echo "  qdrant     http://localhost:6333/dashboard"
	@echo "  minio      http://localhost:9001    (ragent / ragentragent)"

down: ## Stop everything, keep data
	docker compose --profile app down

stop-app: ## Stop only the app containers, leave infra running
	docker compose --profile app stop api worker-parse worker-convert worker-enrich web

reset: ## Stop and destroy all data. Next start reprocesses from scratch.
	docker compose --profile app down -v

logs: ## Tail everything (make logs S=worker-parse for one service)
	docker compose --profile app logs -f $(S)

# ---------------------------------------------------------------- native app

venv: ## Create the local virtualenv
	python -m venv $(VENV)

install: venv ## Install the package and dev tooling into the venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

api: ## Run the API natively with hot reload
	$(PY) -m uvicorn ragent.api.main:app --reload --port 8000

worker: ## Run every stage except convert natively, reloading on save
	$(PY) -m watchfiles --filter python \n	  "$(PY) -m ragent.workers.run --exclude convert" ragent

web: ## Run the web UI natively
	cd web && npm install && npm run dev

worker-convert: ## Run the convert worker in Docker (it needs LibreOffice)
	docker compose --profile app up -d --build worker-convert

# ---------------------------------------------------------------- data

seed: ## Ingest the demo corpus through the API
	$(PY) -m scripts.seed_edgar

seed-local: ## Same, but skip the EDGAR downloads
	$(PY) -m scripts.seed_edgar --local

eval: ## Run the golden set and print the scorecard
	$(PY) -m evals.harness.run

bench: ## Head-to-head every chunking strategy
	$(PY) -m evals.harness.bench --strategies all

mcp: ## Print the MCP server config for Claude Desktop / Claude Code
	$(PY) -m ragent.mcp.print_config

psql: ## Open a shell on the database
	docker compose exec postgres psql -U ragent -d ragent

# ---------------------------------------------------------------- quality

fmt: ## Format
	$(PY) -m ruff format ragent tests scripts
	$(PY) -m ruff check --fix ragent tests scripts

lint: ## Lint and typecheck
	$(PY) -m ruff check ragent tests scripts
	$(PY) -m ruff format --check ragent tests scripts

test: ## Run the test suite
	$(PY) -m pytest -q
