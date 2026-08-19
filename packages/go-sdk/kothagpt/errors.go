package kothagpt

import "fmt"

// Error is the base error type returned by the SDK.
type Error struct {
	StatusCode int
	Message    string
	Body       []byte
}

func (e *Error) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("kothagpt: %s (status %d)", e.Message, e.StatusCode)
	}
	return fmt.Sprintf("kothagpt: request failed (status %d)", e.StatusCode)
}

// NewError builds an Error from an HTTP status and body.
func NewError(statusCode int, body []byte) *Error {
	msg := fmt.Sprintf("request failed with status %d", statusCode)
	return &Error{StatusCode: statusCode, Message: msg, Body: body}
}
