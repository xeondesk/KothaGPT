import { checkStatus } from "./errors.js";

/** Parse a `text/event-stream` response body into JSON payloads. */
export async function* sse<T = unknown>(response: Response): AsyncGenerator<T> {
  if (!response.body) throw new Error("Response has no body stream");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const dataLines = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart());
      for (const data of dataLines) {
        if (data === "[DONE]") return;
        yield JSON.parse(data) as T;
      }
    }
  }
}