.DEFAULT_GOAL := help
.PHONY: help install serve dev test lint format eval-offline eval-real gold free-models docker-build docker-up clean

CONFIG ?= configs/eval.yaml

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies incl. dev tools and the `ner` extra (CPU torch + GLiNER)
	uv sync --all-extras

serve:  ## Gateway + dashboard on http://localhost:8000 (fake upstream unless configured)
	uv run pii-shield serve --port 8000

dev:  ## Same as serve, with auto-reload
	uv run pii-shield serve --port 8000 --reload

test:  ## Test-suite (no API keys; NER tests run only if the GLiNER model is cached)
	uv run pytest

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

gold:  ## Rebuild data/gold/gold.jsonl from the hand-written markup in data/gold/source/
	uv run pii-shield gold build

eval-offline:  ## Detection (patterns, patterns+NER, NER model table) and leak test: no API calls
	uv run pii-shield eval detection -c $(CONFIG) --only patterns,patterns+ner,ner-models
	uv run pii-shield eval leak -c $(CONFIG)

eval-real:  ## LLM-detector comparison and utility test with free OpenRouter models (needs OPENROUTER_API_KEY)
	uv run pii-shield eval detection -c $(CONFIG) --only patterns+ner+llm
	uv run pii-shield eval utility -c $(CONFIG)

free-models:  ## List free OpenRouter models and smoke-test three of them
	uv run pii-shield models free --smoke 3

docker-build:  ## Build the gateway image (NER runtime included, model downloaded on first start)
	docker compose build

docker-up:  ## Gateway + Redis in Docker on http://localhost:8000
	docker compose up --build

clean:  ## Remove caches and the local file vault (keeps results/)
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache .hypothesis
