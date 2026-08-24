.DEFAULT_GOAL := help
.PHONY: help up down logs seed eval bench fmt lint test psql reset mcp

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Bring the whole stack up and wait until it is actually usable
	docker compose up -d --wait
	@echo ""
	@echo "  web        http://localhost:3000"
	@echo "  api docs   http://localhost:8000/docs"
	@echo "  rabbitmq   http://localhost:15672   (ragent / ragent)"
	@echo "  qdrant     http://localhost:6333/dashboard"
	@echo "  minio      http://localhost:9001    (ragent / ragentragent)"
	@echo "  traces     http://localhost:16686"

down: ## Stop the stack, keep volumes
	docker compose down

reset: ## Stop and destroy all data. Next `up` reprocesses from scratch.
	docker compose down -v

logs: ## Tail everything (make logs S=worker-parse for one service)
	docker compose logs -f $(S)

seed: ## Download the demo EDGAR corpus and push it through the pipeline
	docker compose exec api python -m scripts.seed_edgar

eval: ## Run the golden set against the current config and print the scorecard
	docker compose exec api python -m evals.harness.run

bench: ## Head-to-head every chunking strategy, write the comparison chart
	docker compose exec api python -m evals.harness.bench --strategies all

mcp: ## Print the MCP server config to paste into Claude Desktop / Claude Code
	@python -m ragent.mcp.print_config

psql: ## Open a shell on the database
	docker compose exec postgres psql -U ragent -d ragent

fmt: ## Format
	ruff format ragent evals scripts
	ruff check --fix ragent evals scripts

lint: ## Lint and typecheck
	ruff check ragent evals scripts
	mypy ragent

test: ## Run the test suite
	pytest -q
