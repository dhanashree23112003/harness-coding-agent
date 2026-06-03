.PHONY: install test run

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

run:
	python -m agent.main
