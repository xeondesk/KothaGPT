package kothagpt

import (
	"context"
	"encoding/json"
	"fmt"

	"nhooyr.io/websocket"
)

// WebSocketClient is a JSON-over-WebSocket client for the /v1/ws endpoint.
type WebSocketClient struct {
	conn   *websocket.Conn
	nextID int
}

// DialWebSocket connects to the /v1/ws endpoint of baseURL (e.g. ws://localhost:8000).
func DialWebSocket(ctx context.Context, baseURL string) (*WebSocketClient, error) {
	conn, _, err := websocket.Dial(ctx, trimRightSlash(baseURL)+"/v1/ws", nil)
	if err != nil {
		return nil, err
	}
	return &WebSocketClient{conn: conn}, nil
}

func trimRightSlash(s string) string {
	if len(s) > 0 && s[len(s)-1] == '/' {
		return s[:len(s)-1]
	}
	return s
}

// Close closes the underlying connection.
func (w *WebSocketClient) Close() error {
	return w.conn.Close(websocket.StatusNormalClosure, "closing")
}

type wsEnvelope struct {
	ID      string         `json:"id"`
	Type    string         `json:"type"`
	Payload map[string]any `json:"payload"`
}

// Chat sends a chat request over the socket and returns the completion.
func (w *WebSocketClient) Chat(ctx context.Context, messages []Message) (*ChatCompletion, error) {
	payload, err := w.request(ctx, "chat", map[string]any{"messages": messages})
	if err != nil {
		return nil, err
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	var completion ChatCompletion
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, err
	}
	return &completion, nil
}

// Embed sends an embedding request over the socket.
func (w *WebSocketClient) Embed(ctx context.Context, input []string) (*EmbeddingResponse, error) {
	payload, err := w.request(ctx, "embed", map[string]any{"input": input})
	if err != nil {
		return nil, err
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	var out EmbeddingResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// AgentsCreate registers an agent over the socket.
func (w *WebSocketClient) AgentsCreate(ctx context.Context, spec AgentSpec) (*Agent, error) {
	payload, err := w.request(ctx, "agents.create", map[string]any{
		"name":         spec.Name,
		"description":  spec.Description,
		"instructions": spec.Instructions,
		"model":        spec.Model,
		"tools":        spec.Tools,
	})
	if err != nil {
		return nil, err
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	var agent Agent
	if err := json.Unmarshal(raw, &agent); err != nil {
		return nil, err
	}
	return &agent, nil
}

func (w *WebSocketClient) request(ctx context.Context, type_ string, payload map[string]any) (map[string]any, error) {
	w.nextID++
	id := fmt.Sprintf("%d", w.nextID)
	envelope := wsEnvelope{ID: id, Type: type_, Payload: payload}
	data, err := json.Marshal(envelope)
	if err != nil {
		return nil, err
	}
	if err := w.conn.Write(ctx, websocket.MessageText, data); err != nil {
		return nil, err
	}
	_, raw, err := w.conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	var reply wsEnvelope
	if err := json.Unmarshal(raw, &reply); err != nil {
		return nil, err
	}
	if reply.Type == "error" {
		return nil, NewError(0, raw)
	}
	return reply.Payload, nil
}
