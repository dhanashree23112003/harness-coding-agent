.PHONY: install test test-integration eval-retrieval run

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

eval-retrieval:
	python -m evals.retrieval.eval_recall

run:
	python -m agent.main
