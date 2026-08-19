use std::pin::Pin;

use futures_util::{Stream, StreamExt, TryStreamExt};
use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

use crate::error::{api_error, Error, Result};
use crate::types::{
    Agent, AgentRun, AgentSpec, AgentStreamEvent, ChatChunk, ChatCompletion, ChatCompletionRequest,
    EmbeddingResponse, Model, RerankResponse, Tool,
};

const DEFAULT_BASE_URL: &str = "http://localhost:8000";

/// Async client for the Kotha GPT API.
pub struct KothaGPT {
    base_url: String,
    client: reqwest::Client,
}

impl KothaGPT {
    pub fn new() -> Result<Self> {
        Self::with_base_url(DEFAULT_BASE_URL)
    }

    pub fn with_base_url(base_url: &str) -> Result<Self> {
        Ok(KothaGPT {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        })
    }

    pub fn with_api_key(base_url: &str, api_key: &str) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        let value = format!("Bearer {api_key}");
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| Error::Other(format!("invalid api key header: {e}")))?,
        );
        Ok(KothaGPT {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .default_headers(headers)
                .build()?,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl serde::Serialize,
    ) -> Result<T> {
        let response = self.client.post(self.url(path)).json(body).send().await?;
        self.decode(response).await
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let response = self.client.get(self.url(path)).send().await?;
        self.decode(response).await
    }

    async fn decode<T: DeserializeOwned>(&self, response: reqwest::Response) -> Result<T> {
        let status = response.status();
        let body = response.bytes().await?;
        if !status.is_success() {
            return Err(api_error(status.as_u16(), &String::from_utf8_lossy(&body)));
        }
        Ok(serde_json::from_slice(&body)?)
    }

    // ---- chat ----

    pub async fn chat(&self, request: ChatCompletionRequest) -> Result<ChatCompletion> {
        self.post_json("/v1/chat/completions", &request).await
    }

    /// Streams a chat completion. Returns a stream of chunks.
    pub async fn chat_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk>> + Send>>> {
        request.stream = true;
        let response = self
            .client
            .post(self.url("/v1/chat/completions"))
            .json(&request)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(api_error(status.as_u16(), &body));
        }
        Ok(sse_stream(response, |data| {
            serde_json::from_str::<ChatChunk>(data)
        }))
    }

    // ---- embeddings ----

    pub async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse> {
        self.embed_with_model("kothagpt-embed", input).await
    }

    pub async fn embed_with_model(
        &self,
        model: &str,
        input: Vec<String>,
    ) -> Result<EmbeddingResponse> {
        let payload = serde_json::json!({ "model": model, "input": input });
        self.post_json("/v1/embeddings", &payload).await
    }

    // ---- rerank ----

    pub async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<RerankResponse> {
        let payload = serde_json::json!({ "model": "kothagpt-rerank", "query": query, "documents": documents });
        self.post_json("/v1/rerank", &payload).await
    }

    // ---- models ----

    pub async fn models(&self) -> Result<Vec<Model>> {
        #[derive(serde::Deserialize)]
        struct ModelList {
            data: Vec<Model>,
        }
        let list: ModelList = self.get_json("/v1/models").await?;
        Ok(list.data)
    }

    // ---- tools ----

    pub async fn tools(&self) -> Result<Vec<Tool>> {
        #[derive(serde::Deserialize)]
        struct ToolList {
            data: Vec<Tool>,
        }
        let list: ToolList = self.get_json("/v1/tools").await?;
        Ok(list.data)
    }

    pub async fn invoke_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        #[derive(serde::Deserialize)]
        struct ToolResult {
            result: serde_json::Value,
        }
        let payload = serde_json::json!({ "name": name, "arguments": arguments });
        let out: ToolResult = self
            .post_json(&format!("/v1/tools/{name}/invoke"), &payload)
            .await?;
        Ok(out.result)
    }

    // ---- agents ----

    pub async fn create_agent(&self, spec: AgentSpec) -> Result<Agent> {
        self.post_json("/v1/agents", &spec).await
    }

    pub async fn get_agent(&self, agent_id: &str) -> Result<Agent> {
        self.get_json(&format!("/v1/agents/{agent_id}")).await
    }

    pub async fn list_agents(&self) -> Result<Vec<Agent>> {
        #[derive(serde::Deserialize)]
        struct AgentList {
            data: Vec<Agent>,
        }
        let list: AgentList = self.get_json("/v1/agents").await?;
        Ok(list.data)
    }

    pub async fn delete_agent(&self, agent_id: &str) -> Result<()> {
        let response = self
            .client
            .delete(self.url(&format!("/v1/agents/{agent_id}")))
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(api_error(status.as_u16(), &body));
        }
        Ok(())
    }

    pub async fn run_agent(&self, agent_id: &str, message: &str) -> Result<AgentRun> {
        let payload = serde_json::json!({ "message": message });
        self.post_json(&format!("/v1/agents/{agent_id}/runs"), &payload)
            .await
    }

    pub async fn get_run(&self, agent_id: &str, run_id: &str) -> Result<AgentRun> {
        self.get_json(&format!("/v1/agents/{agent_id}/runs/{run_id}"))
            .await
    }

    /// Streams an agent run, yielding events.
    pub async fn agent_stream(
        &self,
        agent_id: &str,
        message: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<AgentStreamEvent>> + Send>>> {
        let payload = serde_json::json!({ "message": message });
        let response = self
            .client
            .post(self.url(&format!("/v1/agents/{agent_id}/runs/stream")))
            .json(&payload)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(api_error(status.as_u16(), &body));
        }
        Ok(sse_stream(response, |data| {
            serde_json::from_str::<AgentStreamEvent>(data)
        }))
    }
}

/// Builds a stream of SSE `data:` payloads parsed as `T`, ending at `[DONE]`.
fn sse_stream<T: DeserializeOwned>(
    response: reqwest::Response,
    parse: impl Fn(&str) -> serde_json::Result<T> + Send + Sync + 'static,
) -> Pin<Box<dyn Stream<Item = Result<T>> + Send>> {
    let byte_stream = response.bytes_stream().map_err(std::io::Error::other);
    let reader = StreamReader::new(byte_stream);
    let lines = FramedRead::new(reader, LinesCodec::new());
    let parse = std::sync::Arc::new(parse);
    let parsed = lines.filter_map(move |line| {
        let parse = parse.clone();
        Box::pin(async move {
            let line = match line {
                Ok(line) => line,
                Err(e) => return Some(Err(Error::Other(format!("sse line error: {e}")))),
            };
            let line = line.trim();
            if !line.starts_with("data:") {
                return None;
            }
            let data = line[5..].trim_start();
            if data == "[DONE]" {
                return None;
            }
            match parse(data) {
                Ok(value) => Some(Ok(value)),
                Err(e) => Some(Err(Error::Json(e))),
            }
        })
    });
    Box::pin(parsed)
}

/// Convenience for awaiting the status of non-JSON responses.
pub async fn ensure_success(status: StatusCode, body: &str) -> Result<()> {
    if status.is_success() {
        Ok(())
    } else {
        Err(api_error(status.as_u16(), body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_trailing_slash() {
        let client = KothaGPT::with_base_url("http://localhost:8000/").unwrap();
        assert_eq!(client.base_url, "http://localhost:8000");
    }
}
