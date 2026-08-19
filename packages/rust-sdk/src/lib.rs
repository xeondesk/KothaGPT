//! Official Kotha GPT Rust SDK.
//!
//! ```no_run
//! use kothagpt::{KothaGPT, types::{ChatCompletionRequest, Message}};
//!
//! #[tokio::main]
//! async fn main() -> kothagpt::Result<()> {
//!     let client = KothaGPT::new()?;
//!     let completion = client
//!         .chat(ChatCompletionRequest::new(vec![Message::user("হ্যালো")]))
//!         .await?;
//!     println!("{}", completion.text());
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod types;
pub mod websocket;

pub use client::KothaGPT;
pub use error::{Error, Result};
pub use websocket::WebSocketClient;
