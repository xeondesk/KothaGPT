use futures_util::{SinkExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;

use crate::error::{Error, Result};
use crate::types::{Agent, AgentRun, ChatCompletion, EmbeddingResponse, Message};

/// JSON-over-WebSocket client for the `/v1/ws` endpoint.
pub struct WebSocketClient {
    write: futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        WsMessage,
    >,
    read: futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
}

#[derive(Serialize)]
struct Envelope<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    type_: &'a str,
    payload: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct EnvelopeReply {
    id: Option<String>,
    #[serde(rename = "type")]
    type_: String,
    payload: serde_json::Value,
}

impl WebSocketClient {
    /// Connects to the `/v1/ws` endpoint of `base_url` (e.g. `ws://localhost:8000`).
    pub async fn connect(base_url: &str) -> Result<Self> {
        let ws_url = format!("{}/v1/ws", base_url.trim_end_matches('/'));
        let (ws, _) = connect_async(ws_url).await?;
        let (write, read) = ws.split();
        Ok(WebSocketClient { write, read })
    }

    async fn request<T: DeserializeOwned>(
        &mut self,
        type_: &str,
        payload: serde_json::Value,
    ) -> Result<T> {
        let id = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let envelope = Envelope {
            id: &id,
            type_,
            payload,
        };
        let text = serde_json::to_string(&envelope)?;
        self.write.send(WsMessage::Text(text)).await?;

        loop {
            let Some(Ok(msg)) = self.read.next().await else {
                return Err(Error::Other("websocket closed".into()));
            };
            let WsMessage::Text(raw) = msg else {
                continue;
            };
            let reply: EnvelopeReply = serde_json::from_str(&raw)?;
            if reply.id.as_deref() == Some(id.as_str()) || reply.id.is_none() {
                if reply.type_ == "error" {
                    return Err(Error::Other(format!("websocket error: {}", reply.payload)));
                }
                return Ok(serde_json::from_value(reply.payload)?);
            }
        }
    }

    pub async fn chat(&mut self, messages: Vec<Message>) -> Result<ChatCompletion> {
        self.request("chat", json!({ "messages": messages })).await
    }

    pub async fn embed(&mut self, input: Vec<String>) -> Result<EmbeddingResponse> {
        self.request("embed", json!({ "input": input })).await
    }

    pub async fn agents_create(&mut self, spec: &crate::types::AgentSpec) -> Result<Agent> {
        self.request("agents.create", serde_json::to_value(spec)?)
            .await
    }

    pub async fn agents_run(&mut self, agent_id: &str, message: &str) -> Result<AgentRun> {
        self.request(
            "agents.run",
            json!({ "agent_id": agent_id, "message": message }),
        )
        .await
    }
}
