# Kotha GPT Rust SDK

Official Rust client for the [Kotha GPT](https://github.com/khulnasoft/KothaGPT) API.
Built on `reqwest`, `tokio`, and `tokio-tungstenite`.

## Install

```toml
[dependencies]
kothagpt = { path = "packages/rust-sdk" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
futures-util = "0.3"
```

## Quick start

```rust
use kothagpt::{KothaGPT, types::{ChatCompletionRequest, Message}};

#[tokio::main]
async fn main() -> kothagpt::Result<()> {
    let client = KothaGPT::new()?;
    let completion = client
        .chat(ChatCompletionRequest::new(vec![Message::user("বাংলায় একটি ছোট গল্প বলো")]))
        .await?;
    println!("{}", completion.text());
    Ok(())
}
```

## Streaming

```rust
use futures_util::StreamExt;

let mut stream = client
    .chat_stream(ChatCompletionRequest::new(vec![Message::user("একটি গল্প বলো")]))
    .await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?.delta_text());
}
```

## Embeddings

```rust
let response = client.embed(vec!["বাংলা".into(), "বাংলাদেশ".into()]).await?;
assert_eq!(response.data[0].embedding.len(), 256);
```

## Reranking

```rust
let response = client.rerank("বাংলা ভাষা", vec!["রান্না".into(), "বাংলা ব্যাকরণ".into()]).await?;
```

## Tools

```rust
let tools = client.tools().await?;
let result = client
    .invoke_tool("calculator", serde_json::json!({ "expression": "(2 + 3) * 4" }))
    .await?;
```

## Agents

```rust
use kothagpt::types::AgentSpec;

let agent = client
    .create_agent(AgentSpec {
        name: "research-assistant".into(),
        description: None,
        instructions: None,
        model: "kothagpt".into(),
        tools: vec![],
        temperature: None,
    })
    .await?;
let run = client.run_agent(&agent.id, "বাংলার রাজধানী কোথায়?").await?;
println!("{}", run.output.unwrap_or_default());
```

## WebSocket

```rust
use kothagpt::{WebSocketClient, types::Message};

let mut ws = WebSocketClient::connect("ws://localhost:8000").await?;
let completion = ws.chat(vec![Message::user("হ্যালো")]).await?;
```

## Configuration

| Environment variable | Used for       |
| -------------------- | -------------- |
| `KOTHAGPT_API_URL`   | API base URL   |

All fallible operations return `kothagpt::Result<T>`; API errors carry an HTTP
status code and response body (`Error::Api { status, body }`).