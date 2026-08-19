use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum Error {
    Http(reqwest::Error),
    Api { status: u16, body: String },
    Json(serde_json::Error),
    Io(std::io::Error),
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Http(e) => write!(f, "http error: {e}"),
            Error::Api { status, body } => write!(f, "api error (status {status}): {body}"),
            Error::Json(e) => write!(f, "json error: {e}"),
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Http(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(e: tokio_tungstenite::tungstenite::Error) -> Self {
        Error::Other(format!("websocket error: {e}"))
    }
}

impl From<url::ParseError> for Error {
    fn from(e: url::ParseError) -> Self {
        Error::Other(format!("url error: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorBody {
    pub detail: Option<serde_json::Value>,
}

pub fn api_error(status: u16, body: &str) -> Error {
    Error::Api {
        status,
        body: body.to_string(),
    }
}
