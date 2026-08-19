# Kotha GPT Go SDK

Official Go client for the [Kotha GPT](https://github.com/khulnasoft/KothaGPT) API.
Uses the standard library `net/http` for REST calls plus `nhooyr.io/websocket` for
WebSocket support.

## Install

```bash
go get kothagpt.dev/sdk/kothagpt
```

## Quick start

```go
package main

import (
	"context"
	"fmt"

	"kothagpt.dev/sdk/kothagpt"
)

func main() {
	client := kothagpt.NewClient("http://localhost:8000")
	completion, err := client.Chat.Create(context.Background(), kothagpt.ChatCompletionRequest{
		Messages: []kothagpt.Message{{Role: kothagpt.RoleUser, Content: "বাংলায় একটি ছোট গল্প বলো"}},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(completion.Text())
}
```

## Streaming

```go
stream, err := client.Chat.Stream(ctx, kothagpt.ChatCompletionRequest{
	Messages: []kothagpt.Message{{Role: kothagpt.RoleUser, Content: "একটি গল্প বলো"}},
})
if err != nil {
	panic(err)
}
defer stream.Stop()
for {
	chunk, err := stream.Next()
	if err == io.EOF {
		break
	}
	if err != nil {
		panic(err)
	}
	fmt.Print(chunk.Choices[0].Delta["content"])
}
```

## Embeddings

```go
resp, err := client.Embeddings.Create(ctx, []string{"বাংলা", "বাংলাদেশ"})
fmt.Println(len(resp.Data[0].Embedding)) // 256
```

## Reranking

```go
resp, err := client.Rerank.Create(ctx, "বাংলা ভাষা", []string{"রান্না", "বাংলা ব্যাকরণ"})
```

## Tools

```go
tools, _ := client.Tools.List(ctx)
result, _ := client.Tools.Invoke(ctx, "calculator", map[string]any{"expression": "(2 + 3) * 4"})
```

## Agents

```go
agent, _ := client.Agents.Create(ctx, kothagpt.AgentSpec{Name: "research-assistant"})
run, _ := client.Agents.Run(ctx, agent.ID, "বাংলার রাজধানী কোথায়?")
fmt.Println(*run.Output)

_ = client.Agents.Stream(ctx, agent.ID, "হ্যালো", func(e *kothagpt.AgentStreamEvent) error {
	fmt.Println(e.Event)
	return nil
})
```

## WebSocket

```go
ws, err := kothagpt.DialWebSocket(ctx, "ws://localhost:8000")
defer ws.Close()
completion, err := ws.Chat(ctx, []kothagpt.Message{{Role: kothagpt.RoleUser, Content: "হ্যালো"}})
```

## Configuration

| Environment variable | Used for       |
| -------------------- | -------------- |
| `KOTHAGPT_API_URL`   | API base URL   |
| `KOTHAGPT_API_KEY`   | Bearer API key |

All errors implement `error` via the `*kothagpt.Error` type carrying `StatusCode`.