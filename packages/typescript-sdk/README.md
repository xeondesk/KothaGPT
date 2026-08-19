# Kotha GPT TypeScript SDK

Official TypeScript client for the [Kotha GPT](https://github.com/khulnasoft/KothaGPT) API.
Works in browsers and Node.js (>= 18) with zero runtime dependencies.

## Install

```bash
npm install @kothagpt/typescript-sdk
# or
pnpm add @kothagpt/typescript-sdk
```

## Quick start

```typescript
import { KothaGPT } from "@kothagpt/typescript-sdk";

const client = new KothaGPT({ baseURL: "http://localhost:8000", apiKey: "sk-..." });

const completion = await client.chat.create({
  messages: [{ role: "user", content: "বাংলায় একটি ছোট গল্প বলো" }],
});
console.log(completion.choices[0].message.content);
```

## Streaming

```typescript
for await (const chunk of client.chat.stream({
  messages: [{ role: "user", content: "একটি গল্প বলো" }],
})) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

## Embeddings

```typescript
const response = await client.embeddings.create(["বাংলা", "বাংলাদেশ"]);
console.log(response.data[0].embedding.length); // 256
```

## Reranking

```typescript
const response = await client.rerank.create("বাংলা ভাষা", ["রান্না", "বাংলা ব্যাকরণ"], { top_n: 1 });
```

## Tools

```typescript
const tools = await client.tools.list();
const result = await client.tools.invoke("calculator", { expression: "(2 + 3) * 4" });
```

## Agents

```typescript
const agent = await client.agents.create({
  name: "research-assistant",
  instructions: "সংক্ষিপ্ত উত্তর দাও।",
});
const run = await client.agents.run(agent.id, "বাংলার রাজধানী কোথায়?");
console.log(run.output);

for await (const event of client.agents.stream(agent.id, "হ্যালো")) {
  if (event.event === "run.delta") process.stdout.write(String(event.delta ?? ""));
}
```

## WebSocket

```typescript
import { KothaGPTWebSocket } from "@kothagpt/typescript-sdk";

const ws = new KothaGPTWebSocket("ws://localhost:8000");
await ws.connect();
const completion = await ws.chat([{ role: "user", content: "হ্যালো" }]);
ws.close();
```

## Configuration

| Environment variable | Used for       |
| -------------------- | -------------- |
| `KOTHAGPT_API_URL`   | API base URL   |
| `KOTHAGPT_API_KEY`   | Bearer API key |

Errors subclass `KothaGPTError` (`APIError`, `AuthenticationError`, `NotFoundError`).