package kothagpt

import (
	"context"
	"net/http"
)

// AgentsResource exposes agent management and execution.
type AgentsResource struct{ c *Client }

// Create registers a new agent from the given spec.
func (r *AgentsResource) Create(ctx context.Context, spec AgentSpec) (*Agent, error) {
	var out Agent
	if err := r.c.doJSON(ctx, http.MethodPost, "/v1/agents", spec, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Get fetches an agent by ID.
func (r *AgentsResource) Get(ctx context.Context, id string) (*Agent, error) {
	var out Agent
	if err := r.c.doJSON(ctx, http.MethodGet, "/v1/agents/"+id, nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// List returns all agents.
func (r *AgentsResource) List(ctx context.Context) ([]Agent, error) {
	var out struct {
		Data []Agent `json:"data"`
	}
	if err := r.c.doJSON(ctx, http.MethodGet, "/v1/agents", nil, &out); err != nil {
		return nil, err
	}
	return out.Data, nil
}

// Delete removes an agent by ID.
func (r *AgentsResource) Delete(ctx context.Context, id string) error {
	return r.c.doJSON(ctx, http.MethodDelete, "/v1/agents/"+id, nil, nil)
}

// Run executes a single-turn agent run.
func (r *AgentsResource) Run(ctx context.Context, id, message string) (*AgentRun, error) {
	var out AgentRun
	payload := map[string]string{"message": message}
	if err := r.c.doJSON(ctx, http.MethodPost, "/v1/agents/"+id+"/runs", payload, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetRun fetches a run by ID.
func (r *AgentsResource) GetRun(ctx context.Context, agentID, runID string) (*AgentRun, error) {
	var out AgentRun
	if err := r.c.doJSON(ctx, http.MethodGet, "/v1/agents/"+agentID+"/runs/"+runID, nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Stream executes a streaming agent run, yielding events.
func (r *AgentsResource) Stream(ctx context.Context, id, message string, emit func(*AgentStreamEvent) error) error {
	resp, err := r.c.streamRequest(ctx, "/v1/agents/"+id+"/runs/stream", map[string]string{"message": message})
	if err != nil {
		return err
	}
	return scanSSE(resp, func(payload []byte) error {
		var event AgentStreamEvent
		if err := jsonUnmarshal(payload, &event); err != nil {
			return err
		}
		return emit(&event)
	})
}
