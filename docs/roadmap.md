# Roadmap

Master checklist (auto-verified coverage): `docs/checklist.md`

## Phase 0 — Foundation (`docs/foundation-plan.md`)
- [ ] AI-এর মূল উদ্দেশ্য নির্ধারণ
- [ ] Language scope (bn/en/multilingual) ঠিক করা
- [ ] Chat / Coding / Agent / Research AI নির্বাচন
- [ ] Target users নির্ধারণ
- [ ] MVP feature list তৈরি
- [ ] Model size নির্ধারণ — 1B / 3B / 7B / 14B+
- [ ] License ও open-source strategy নির্ধারণ
- [ ] Project name ও repository তৈরি
- [ ] Repository bootstrap
- [ ] CI
- [ ] coding standards
- [ ] configuration system

## Phase 1 — Data (`docs/dataset-pipeline-plan.md`)
- [x] corpus ingestion
- [x] normalization
- [x] filtering
- [x] deduplication
- [x] dataset versioning

## Phase 2 — Language
- [x] tokenizer
- [x] Bangla benchmarks
- [x] vocabulary optimization

## Phase 3 — Model (`docs/base-model-plan.md` → `docs/pretraining-plan.md`)
- [x] architecture (decoder-only, RoPE, SwiGLU, RMSNorm)
- [ ] small-model experiment
- [~] pretraining framework (config, dataset, loop, resume; real run pending)
- [x] checkpointing

## Phase 4 — Alignment
- [ ] SFT (`docs/sft-plan.md`)
- [ ] preference tuning (`docs/preference-plan.md`)
- [ ] safety (part of `docs/preference-plan.md`)

## Phase 5 — Runtime (`docs/runtime-plan.md`)
- [ ] inference API
- [ ] streaming
- [ ] quantization
- [ ] model registry

## Phase 6 — Intelligence
- [ ] RAG (`docs/rag-plan.md`)
- [ ] tools (part of `docs/agent-plan.md`)
- [ ] memory (part of `docs/agent-plan.md`)
- [ ] agents (`docs/agent-plan.md`)

## Phase 6b — Context Engineering (`docs/context-engineering-plan.md`)
- [ ] system prompt + instruction hierarchy + few-shot + templates
- [ ] context window + packing + selective context
- [ ] retrieval assembly + history compression + summarization
- [ ] tool/structured-output/injection-safe formatting
- [ ] context eval + prompt registry

## Phase 7 — Platform (`docs/web-app-plan.md` + `docs/sdk.md`)
- [ ] web app
- [ ] SDKs (`docs/sdk-plan.md`)
- [ ] playground
- [ ] developer portal

## Phase 7b — Ecosystem (`docs/ecosystem-plan.md`)
- [ ] hubs (model, dataset) + marketplaces (agent, tool) + prompt library
- [ ] evaluation hub + community
- [ ] fine-tuning platform + hosted inference
- [ ] enterprise API + local/offline AI
- [ ] open-source repositories

## Phase 8 — Evaluation (`docs/eval-plan.md`)
- [ ] benchmark families (general, bn, en, coding, reasoning, math, RAG, agent, hallucination, safety)
- [ ] latency + token-efficiency
- [ ] human evaluation
- [ ] regression gate

## Phase 9 — Infrastructure (`docs/infra-plan.md`)
- [ ] GPU cluster + model serving (LB, autoscaling)
- [ ] data services (Redis, Postgres, object storage, Qdrant, queue)
- [ ] observability (metrics, logs, tracing)
- [ ] cost monitoring
- [ ] disaster recovery

## Phase 10 — Security (`docs/security-plan.md`)
- [ ] prompt injection + tool authorization + sandboxing
- [ ] secrets + auth + rate limiting + encryption
- [ ] PII guard + audit + abuse detection
- [ ] poisoning detection + supply chain + artifact signing
- [ ] red-team testing
