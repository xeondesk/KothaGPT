import { KothaGPT } from "@kothagpt/typescript-sdk";

const client = new KothaGPT({ baseURL: "http://localhost:8000" });

// Chat completion
const completion = await client.chat.create({
  messages: [{ role: "user", content: "বাংলায় একটি ছোট গল্প বলো" }],
});
console.log(completion.choices[0].message.content);

// Streaming
console.log("\n--- streaming ---");
for await (const chunk of client.chat.stream({
  messages: [{ role: "user", content: "একটি কবিতা লেখো" }],
})) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
console.log();

// Embeddings + rerank
const docs = ["বাংলা ভাষা শেখার উপায়", "রান্নার রেসিপি", "ঢাকা শহর"];
const reranked = await client.rerank.create("বাংলা ভাষা", docs, { top_n: 2 });
console.log("\n--- rerank ---");
for (const result of reranked.results) {
  console.log(`[${result.relevance_score.toFixed(3)}] ${result.document}`);
}

// Agent
const agent = await client.agents.create({ name: "ts-helper" });
const run = await client.agents.run(agent.id, "২ + ৩ কত?");
console.log("\n--- agent run ---");
console.log(run.output);

// Tool
const value = await client.tools.invoke<{ value: number }>("calculator", {
  expression: "2 + 3 * 4",
});
console.log("\ncalculator:", value.value);