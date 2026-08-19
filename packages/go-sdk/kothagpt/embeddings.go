package kothagpt

import (
	"context"
	"net/http"
)

// EmbeddingsResource exposes the embedding endpoint.
type EmbeddingsResource struct{ c *Client }

// Create embeds one or more strings and returns the response.
func (r *EmbeddingsResource) Create(ctx context.Context, input []string) (*EmbeddingResponse, error) {
	return r.CreateWithModel(ctx, "kothagpt-embed", input)
}

// CreateWithModel embeds strings using the given model.
func (r *EmbeddingsResource) CreateWithModel(ctx context.Context, model string, input []string) (*EmbeddingResponse, error) {
	var out EmbeddingResponse
	payload := map[string]any{"model": model, "input": input}
	if err := r.c.doJSON(ctx, http.MethodPost, "/v1/embeddings", payload, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// RerankResource exposes the reranking endpoint.
type RerankResource struct{ c *Client }

// Create reranks documents against a query and returns scored results.
func (r *RerankResource) Create(ctx context.Context, query string, documents []string) (*RerankResponse, error) {
	return r.CreateWithModel(ctx, "kothagpt-rerank", query, documents, 0)
}

// CreateWithModel reranks with a specific model and optional top_n (0 = all).
func (r *RerankResource) CreateWithModel(ctx context.Context, model, query string, documents []string, topN int) (*RerankResponse, error) {
	var out RerankResponse
	payload := map[string]any{"model": model, "query": query, "documents": documents}
	if topN > 0 {
		payload["top_n"] = topN
	}
	if err := r.c.doJSON(ctx, http.MethodPost, "/v1/rerank", payload, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
