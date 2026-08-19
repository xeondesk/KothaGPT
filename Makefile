bootstrap:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r services/api/requirements.txt
	corepack enable
	pnpm install

dev:
	uvicorn services.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests

lint:
	ruff check .
	cargo fmt --check
	pnpm lint

format:
	ruff format .
	cargo fmt
	pnpm format

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
