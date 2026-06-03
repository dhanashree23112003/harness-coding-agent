.PHONY: install test test-integration run

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

run:
	python -m agent.main
