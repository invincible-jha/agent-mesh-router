.PHONY: install test lint typecheck format security ci clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

security:
	pip-audit

ci: lint typecheck test security

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
