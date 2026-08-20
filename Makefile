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
	.venv/bin/python -m pytest tests

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

data-validate:
	@echo "Running data validation checks..."
	python -m data.pipeline.cli version

tokenizer-build:
	@echo "Building tokenizer artifacts..."
	python -m ml.tokenizer.cli train \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--out ml/tokenizer/artifacts

# CI smoke: build a tiny dataset + tokenizer from committed fixtures so the
# data-validate / tokenizer-check gates below have artifacts to validate.
ci-data-build:
	python -m data.pipeline.cli run \
		--raw-dir tests/fixtures/raw \
		--out-root data/processed \
		--version-label ci-smoke \
		--min-chars 1 --max-chars 1000000 --min-words 1 \
		--no-gzip

ci-tokenizer-build:
	python -m ml.tokenizer.cli train \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--algorithm bpe --vocab-size 2000 --min-frequency 1 \
		--out ml/tokenizer/artifacts/best

tokenizer-check:
	@if [ -f ml/tokenizer/artifacts/best/tokenizer.json ]; then \
		echo "tokenizer artifact OK: ml/tokenizer/artifacts/best/tokenizer.json"; \
	else \
		echo "ERROR: tokenizer artifact missing. Run 'make ci-tokenizer-build' (or 'make tokenizer-build' on real data) first."; \
		exit 1; \
	fi

# WS-7 efficiency gate: fails the build when the frozen tokenizer violates the
# threshold set (unk < 0.5% on dev, decode fidelity 100%, tokens/char budget).
tokenizer-bench:
	python -m ml.tokenizer.cli benchmark \
		--tokenizer ml/tokenizer/artifacts/best/tokenizer.json \
		--out ml/tokenizer/artifacts/benchmark.json \
		--gate

# WS-9: run the Bangla benchmark suite against the mock backend and write a
# dated JSON report + markdown report under evals/results/.
eval-bangla:
	python -m evals.run --suite evals/suites/bangla.yaml

# WS-1/WS-6: retrain on the normalized corpus and freeze tokenizer + vocab
# (artifacts/best/, vocab/, DECISION.md). On real data use the target vocab
# size; CI uses a small size so the artifact is cheap to produce.
tokenizer-freeze:
	python -m ml.tokenizer.cli freeze \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--algorithm bpe --vocab-size 2000 --min-frequency 1 --gate \
		--out ml/tokenizer

serve-proto:
	uvicorn services.api.app:app --host 0.0.0.0 --port 8000

# --- Developer SDK -----------------------------------------------------------

# --- Docker Compose ----------------------------------------------------------

.PHONY: up down logs ps stop restart build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

stop:
	docker compose stop

restart:
	docker compose restart

build:
	docker compose build

# --- Developer SDK -----------------------------------------------------------

.PHONY: sdk-install sdk-test sdk-lint sdk-build

sdk-install:
	.venv/bin/python -m pip install -e packages/python-sdk packages/cli
	cd packages/typescript-sdk && npm install
	cd packages/go-sdk && go mod tidy
	cd packages/rust-sdk && cargo build

sdk-test:
	.venv/bin/python -m pytest tests/test_api.py tests/test_api_sdk.py tests/test_python_sdk.py tests/test_cli.py
	cd packages/typescript-sdk && npx vitest run
	cd packages/go-sdk && go test ./...
	cd packages/rust-sdk && cargo test

sdk-lint:
	.venv/bin/python -m ruff check packages/python-sdk packages/cli tests
	cd packages/typescript-sdk && npm run lint
	cd packages/go-sdk && gofmt -l kothagpt && go vet ./...
	cd packages/rust-sdk && cargo fmt --check && cargo clippy --all-targets

sdk-build:
	cd packages/typescript-sdk && npm run build
	cd packages/go-sdk && go build ./...
	cd packages/rust-sdk && cargo build --release
