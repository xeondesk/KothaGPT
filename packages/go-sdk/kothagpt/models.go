package kothagpt

import (
	"context"
	"net/http"
)

// ModelsResource exposes model listing.
type ModelsResource struct{ c *Client }

// List returns the models available on the server.
func (r *ModelsResource) List(ctx context.Context) ([]Model, error) {
	var out struct {
		Data []Model `json:"data"`
	}
	if err := r.c.doJSON(ctx, http.MethodGet, "/v1/models", nil, &out); err != nil {
		return nil, err
	}
	return out.Data, nil
}

// ToolsResource exposes tool discovery and invocation.
type ToolsResource struct{ c *Client }

// List returns the registered tools.
func (r *ToolsResource) List(ctx context.Context) ([]Tool, error) {
	var out struct {
		Data []Tool `json:"data"`
	}
	if err := r.c.doJSON(ctx, http.MethodGet, "/v1/tools", nil, &out); err != nil {
		return nil, err
	}
	return out.Data, nil
}

// Invoke calls a tool by name with the given arguments and returns its result.
func (r *ToolsResource) Invoke(ctx context.Context, name string, arguments map[string]any) (any, error) {
	var out struct {
		Result any `json:"result"`
	}
	payload := map[string]any{"name": name, "arguments": arguments}
	if err := r.c.doJSON(ctx, http.MethodPost, "/v1/tools/"+name+"/invoke", payload, &out); err != nil {
		return nil, err
	}
	return out.Result, nil
}
