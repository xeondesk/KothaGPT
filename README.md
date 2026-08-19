# Own AI — Bangla-First AI Platform

A monorepo scaffold for building a Bangla-first AI ecosystem:
dataset → tokenizer → pretraining → instruction tuning → evaluation → inference → RAG → agents → SDKs → platform.

## Stack
- Python: training, data, evaluation, orchestration
- Rust: high-performance runtime/components
- TypeScript: web platform and SDK
- PyTorch: model training
- FastAPI: API
- PostgreSQL: metadata
- Redis: cache/queues
- Qdrant/pgvector: vector search
- Next.js: web UI

## Repository layout
- `apps/` user-facing applications
- `services/` backend services
- `packages/` shared SDKs and libraries
- `ml/` model/training stack
- `data/` dataset pipeline
- `evals/` benchmarks and evaluation
- `infra/` deployment/infrastructure
- `docs/` engineering documentation
- `configs/` shared configuration
- `scripts/` developer automation

## Quick start

```bash
cp .env.example .env
make bootstrap
make test
```

Training is intentionally disabled by default. Configure a dataset and GPU environment before running training commands.
