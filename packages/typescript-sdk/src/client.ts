import {
  Agent,
  AgentRun,
  AgentSpec,
  AgentStreamEvent,
  ChatCompletion,
  ChatMessage,
  ChatChunk,
  ChatCompletionRequest,
  EmbeddingResponse,
  Model,
  RerankResponse,
  Tool,
} from "./types.js";
import { checkStatus } from "./errors.js";
import { sse } from "./streaming.js";

export interface KothaGPTOptions {
  baseURL?: string;
  apiKey?: string;
  fetch?: typeof fetch;
}

export class KothaGPT {
  readonly chat: ChatCompletions;
  readonly embeddings: Embeddings;
  readonly rerank: Rerank;
  readonly models: Models;
  readonly tools: Tools;
  readonly agents: Agents;

  private readonly baseURL: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: KothaGPTOptions = {}) {
    this.baseURL = (options.baseURL ?? "http://localhost:8000").replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? fetch;
    this.chat = new ChatCompletions(this);
    this.embeddings = new Embeddings(this);
    this.rerank = new Rerank(this);
    this.models = new Models(this);
    this.tools = new Tools(this);
    this.agents = new Agents(this);
  }

  request(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set("Content-Type", "application/json");
    if (this.apiKey) headers.set("Authorization", `Bearer ${this.apiKey}`);
    return this.fetchImpl(`${this.baseURL}${path}`, { ...init, headers });
  }

  async json<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await checkStatus(await this.request(path, init));
    return (await response.json()) as T;
  }
}

export class ChatCompletions {
  constructor(private readonly client: KothaGPT) {}

  async create(request: ChatCompletionRequest): Promise<ChatCompletion> {
    return this.client.json("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async *stream(request: ChatCompletionRequest): AsyncGenerator<ChatChunk> {
    const response = await checkStatus(
      await this.client.request("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({ ...request, stream: true }),
      }),
    );
    yield* sse<ChatChunk>(response);
  }
}

export class Embeddings {
  constructor(private readonly client: KothaGPT) {}

  async create(input: string | string[], model = "kothagpt-embed"): Promise<EmbeddingResponse> {
    return this.client.json("/v1/embeddings", {
      method: "POST",
      body: JSON.stringify({ model, input }),
    });
  }
}

export class Rerank {
  constructor(private readonly client: KothaGPT) {}

  async create(
    query: string,
    documents: string[],
    options: { model?: string; top_n?: number } = {},
  ): Promise<RerankResponse> {
    return this.client.json("/v1/rerank", {
      method: "POST",
      body: JSON.stringify({ model: options.model ?? "kothagpt-rerank", query, documents, top_n: options.top_n }),
    });
  }
}

export class Models {
  constructor(private readonly client: KothaGPT) {}

  async list(): Promise<Model[]> {
    const data = await this.client.json<{ data: Model[] }>("/v1/models");
    return data.data;
  }
}

export class Tools {
  constructor(private readonly client: KothaGPT) {}

  async list(): Promise<Tool[]> {
    const data = await this.client.json<{ data: Tool[] }>("/v1/tools");
    return data.data;
  }

  async invoke<T = unknown>(name: string, arguments_: Record<string, unknown> = {}): Promise<T> {
    const data = await this.client.json<{ result: T }>(`/v1/tools/${name}/invoke`, {
      method: "POST",
      body: JSON.stringify({ name, arguments: arguments_ }),
    });
    return data.result;
  }
}

export class Agents {
  constructor(private readonly client: KothaGPT) {}

  async create(spec: AgentSpec): Promise<Agent> {
    return this.client.json("/v1/agents", { method: "POST", body: JSON.stringify(spec) });
  }

  async get(id: string): Promise<Agent> {
    return this.client.json(`/v1/agents/${id}`);
  }

  async list(): Promise<Agent[]> {
    const data = await this.client.json<{ data: Agent[] }>("/v1/agents");
    return data.data;
  }

  async delete(id: string): Promise<void> {
    const response = await this.client.request(`/v1/agents/${id}`, { method: "DELETE" });
    await checkStatus(response);
  }

  async run(id: string, message: string): Promise<AgentRun> {
    return this.client.json(`/v1/agents/${id}/runs`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
  }

  async *stream(id: string, message: string): AsyncGenerator<AgentStreamEvent> {
    const response = await checkStatus(
      await this.client.request(`/v1/agents/${id}/runs/stream`, {
        method: "POST",
        body: JSON.stringify({ message }),
      }),
    );
    yield* sse<AgentStreamEvent>(response);
  }

  async getRun(id: string, runId: string): Promise<AgentRun> {
    return this.client.json(`/v1/agents/${id}/runs/${runId}`);
  }
}

export type { ChatMessage } from "./types.js";