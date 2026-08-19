# Developer SDK

The Kotha GPT platform exposes a stable HTTP API (REST + SSE streaming +
WebSocket) consumed by first-party SDKs in multiple languages.

## Quick reference

| Layer          | Location                                             |
| -------------- | ---------------------------------------------------- |
| REST API       | `services/api` (FastAPI, `/v1/*` routes)             |
| Streaming API  | SSE over `POST /v1/chat/completions` (`stream: true`) and `POST /v1/chat/stream` |
| WebSocket API  | `/v1/ws` (JSON request/response envelopes)           |
| Python SDK     | `packages/python-sdk` (`pip install kothagpt`)       |
| TypeScript SDK | `packages/typescript-sdk` (`@kothagpt/typescript-sdk`)|
| Go SDK         | `packages/go-sdk` (`kothagpt.dev/sdk/kothagpt`)      |
| Rust SDK       | `packages/rust-sdk` (`kothagpt` crate)               |
| CLI            | `packages/cli` (`kothagpt` executable)               |
| Examples       | `examples/`                                          |

## API surface

All endpoints live under `/v1` unless noted.

| Method | Path                                   | Description                    |
| ------ | -------------------------------------- | ------------------------------ |
| GET    | `/v1/models`                           | List models                    |
| POST   | `/v1/chat/completions`                 | Chat completion (REST + SSE)   |
| POST   | `/v1/chat/stream`                      | Always-streaming chat (SSE)    |
| POST   | `/v1/chat`                             | Legacy single-message chat     |
| POST   | `/v1/embeddings`                       | Text embeddings                |
| POST   | `/v1/rerank`                           | Query/document reranking       |
| GET    | `/v1/tools`                            | List tools                     |
| GET    | `/v1/tools/{name}`                     | Get a tool                     |
| POST   | `/v1/tools/{name}/invoke`              | Invoke a tool                  |
| GET    | `/v1/agents`                           | List agents                    |
| POST   | `/v1/agents`                           | Create an agent                |
| GET    | `/v1/agents/{id}`                      | Get an agent                   |
| DELETE | `/v1/agents/{id}`                      | Delete an agent                |
| POST   | `/v1/agents/{id}/runs`                 | Run an agent                   |
| POST   | `/v1/agents/{id}/runs/stream`          | Stream an agent run (SSE)      |
| GET    | `/v1/agents/{id}/runs/{run_id}`        | Get a run                      |
| WS     | `/v1/ws`                               | JSON-over-WebSocket gateway    |

## Authentication

Pass a bearer token via `Authorization: Bearer <key>`. If no key is configured
the server accepts unauthenticated requests (development default).

## Swapping backends

The API is backed by a pluggable backend interface (`services/api/core/backend.py`).
The default `mock` backend is deterministic and dependency-free, making every SDK
testable without model weights. Set `KOTHAGPT_BACKEND=real` and register a real
backend to serve production traffic.

## Getting started

```bash
# run the API
make serve-proto

# quick smoke test
curl http://localhost:8000/v1/models

# try the CLI
kothagpt models
kothagpt chat --stream "বাংলায় হ্যালো"
```