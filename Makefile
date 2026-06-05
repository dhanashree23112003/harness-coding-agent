.PHONY: install test test-integration eval eval-retrieval run run-long docker-build

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

eval:
	python -m evals.retrieval.eval_recall

eval-retrieval:
	python -m evals.retrieval.eval_recall

run:
	python -m agent.main

run-long:
	CONTEXT_COMPACT_THRESHOLD=800 python -m agent.main long_horizon

docker-build:
	docker build -t agent .
