bootstrap:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r services/api/requirements.txt
	corepack enable
	pnpm install

dev:
	uvicorn services.api.app:app --reload --host 0.0.0.0 --port 8000

data:
	python -m data.pipeline.cli run

data-check:
	python -m data.pipeline.cli version

tokenizer:
	python -m ml.tokenizer.cli experiments \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--out ml/tokenizer/artifacts

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
