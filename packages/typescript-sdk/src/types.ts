export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ChatMessage {
  role: Role;
  content: string;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface FunctionDefinition {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

export interface Tool {
  type: "function";
  function: FunctionDefinition;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: Tool[];
  tool_choice?: string | Record<string, unknown>;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatChoice {
  index: number;
  message: ChatMessage;
  finish_reason: string | null;
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatChoice[];
  usage: Usage;
}

export interface ChatChunkChoice {
  index: number;
  delta: Partial<ChatMessage>;
  finish_reason: string | null;
}

export interface ChatChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatChunkChoice[];
}

export interface Embedding {
  object: "embedding";
  index: number;
  embedding: number[];
}

export interface EmbeddingResponse {
  object: "list";
  model: string;
  data: Embedding[];
  usage: Usage;
}

export interface RerankResult {
  index: number;
  document: string;
  relevance_score: number;
}

export interface RerankResponse {
  object: "list";
  model: string;
  results: RerankResult[];
}

export interface Model {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  description: string;
  context_window: number;
}

export interface AgentSpec {
  name: string;
  description?: string;
  instructions?: string;
  model?: string;
  tools?: string[];
  temperature?: number;
}

export interface Agent {
  id: string;
  object: "agent";
  name: string;
  description: string | null;
  instructions: string | null;
  model: string;
  tools: string[];
  temperature: number | null;
  created_at: number;
}

export interface AgentRun {
  id: string;
  object: "agent.run";
  agent_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  messages: ChatMessage[];
  output: string | null;
  created_at: number;
  updated_at: number;
}

export interface AgentStreamEvent {
  event: "run.created" | "run.delta" | "run.completed";
  [key: string]: unknown;
}