package kothagpt

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// ChatResource exposes chat completion endpoints.
type ChatResource struct{ c *Client }

// Create sends a chat completion request and returns the full response.
func (r *ChatResource) Create(ctx context.Context, req ChatCompletionRequest) (*ChatCompletion, error) {
	var out ChatCompletion
	if err := r.c.doJSON(ctx, http.MethodPost, "/v1/chat/completions", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Stream sends a chat completion request and returns an iterator of chunks.
// Callers must Stop() the returned stream to release the connection.
func (r *ChatResource) Stream(ctx context.Context, req ChatCompletionRequest) (*ChatStream, error) {
	req.Stream = true
	resp, err := r.c.streamRequest(ctx, "/v1/chat/completions", req)
	if err != nil {
		return nil, err
	}
	return &ChatStream{resp: resp}, nil
}

// ChatStream iterates over streaming chat chunks.
type ChatStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
}

// Next returns the next chunk, or io.EOF when the stream ends.
func (s *ChatStream) Next() (*ChatChunk, error) {
	if s.scanner == nil {
		s.scanner = bufio.NewScanner(s.resp.Body)
		s.scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	}
	for s.scanner.Scan() {
		line := strings.TrimSpace(s.scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return nil, io.EOF
		}
		var chunk ChatChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, err
		}
		return &chunk, nil
	}
	if err := s.scanner.Err(); err != nil {
		return nil, err
	}
	return nil, io.EOF
}

// Stop closes the underlying response body.
func (s *ChatStream) Stop() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}
