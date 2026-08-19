package kothagpt

// Role identifies the sender of a chat message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type FunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

type ChatCompletionRequest struct {
	Model       string    `json:"model,omitempty"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  any       `json:"tool_choice,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatChoice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason *string `json:"finish_reason"`
}

type ChatCompletion struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

// Text joins the content of all assistant choices.
func (c *ChatCompletion) Text() string {
	var out string
	for _, choice := range c.Choices {
		out += choice.Message.Content
	}
	return out
}

type ChatChunkChoice struct {
	Index        int            `json:"index"`
	Delta        map[string]any `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

type ChatChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ChatChunkChoice `json:"choices"`
}

type Embedding struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type EmbeddingResponse struct {
	Object string      `json:"object"`
	Model  string      `json:"model"`
	Data   []Embedding `json:"data"`
	Usage  Usage       `json:"usage"`
}

type RerankResult struct {
	Index          int     `json:"index"`
	Document       string  `json:"document"`
	RelevanceScore float64 `json:"relevance_score"`
}

type RerankResponse struct {
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Results []RerankResult `json:"results"`
}

type Model struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	Created       int64  `json:"created"`
	OwnedBy       string `json:"owned_by"`
	Description   string `json:"description"`
	ContextWindow int    `json:"context_window"`
}

type AgentSpec struct {
	Name         string   `json:"name"`
	Description  string   `json:"description,omitempty"`
	Instructions string   `json:"instructions,omitempty"`
	Model        string   `json:"model,omitempty"`
	Tools        []string `json:"tools,omitempty"`
	Temperature  *float64 `json:"temperature,omitempty"`
}

type Agent struct {
	ID           string   `json:"id"`
	Object       string   `json:"object"`
	Name         string   `json:"name"`
	Description  *string  `json:"description"`
	Instructions *string  `json:"instructions"`
	Model        string   `json:"model"`
	Tools        []string `json:"tools"`
	Temperature  *float64 `json:"temperature"`
	CreatedAt    int64    `json:"created_at"`
}

type AgentRun struct {
	ID        string    `json:"id"`
	Object    string    `json:"object"`
	AgentID   string    `json:"agent_id"`
	Status    string    `json:"status"`
	Messages  []Message `json:"messages"`
	Output    *string   `json:"output"`
	CreatedAt int64     `json:"created_at"`
	UpdatedAt int64     `json:"updated_at"`
}

type AgentStreamEvent struct {
	Event string `json:"event"`
	// Event-specific payload fields are kept raw.
	Raw map[string]any
}

func (e *AgentStreamEvent) UnmarshalJSON(data []byte) error {
	var raw map[string]any
	if err := jsonUnmarshal(data, &raw); err != nil {
		return err
	}
	e.Event, _ = raw["event"].(string)
	delete(raw, "event")
	e.Raw = raw
	return nil
}
