package kothagpt

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client is the Kotha GPT Go SDK client.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	userAgent  string

	Chat       *ChatResource
	Embeddings *EmbeddingsResource
	Rerank     *RerankResource
	Models     *ModelsResource
	Tools      *ToolsResource
	Agents     *AgentsResource
}

// ClientOption configures the Client.
type ClientOption func(*Client)

// WithAPIKey sets the bearer token sent on every request.
func WithAPIKey(key string) ClientOption {
	return func(c *Client) { c.apiKey = key }
}

// WithHTTPClient overrides the underlying HTTP client.
func WithHTTPClient(hc *http.Client) ClientOption {
	return func(c *Client) { c.httpClient = hc }
}

// NewClient constructs a Client pointing at the given base URL.
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 60 * time.Second},
		userAgent:  "kothagpt-go-sdk/0.1.0",
	}
	for _, opt := range opts {
		opt(c)
	}
	c.Chat = &ChatResource{c: c}
	c.Embeddings = &EmbeddingsResource{c: c}
	c.Rerank = &RerankResource{c: c}
	c.Models = &ModelsResource{c: c}
	c.Tools = &ToolsResource{c: c}
	c.Agents = &AgentsResource{c: c}
	return c
}

func (c *Client) doJSON(ctx context.Context, method, path string, body, out any) error {
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return NewError(resp.StatusCode, raw)
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *Client) streamRequest(ctx context.Context, path string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return nil, NewError(resp.StatusCode, raw)
	}
	return resp, nil
}

// scanSSE parses a text/event-stream response and calls emit for each data payload.
// Terminated by the [DONE] sentinel or EOF.
func scanSSE(resp *http.Response, emit func(payload []byte) error) error {
	defer resp.Body.Close()
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return nil
		}
		if err := emit([]byte(data)); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func jsonUnmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}
