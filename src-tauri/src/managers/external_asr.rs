//! External ASR Client for MLX FastAPI
//!
//! This module provides HTTP client functionality to call the MLX FastAPI
//! ASR service instead of running the model locally.

use anyhow::{Context, Result};
use log::{debug, error, info};
use reqwest::blocking::multipart;
use reqwest::blocking::Client;
use std::time::Duration;

/// External ASR Client configuration
pub struct ExternalASRClient {
    base_url: String,
    model: String,
    #[allow(dead_code)]
    client: Client,
}

impl ExternalASRClient {
    /// Create a new External ASR Client with default configuration
    #[allow(dead_code)]
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: "http://127.0.0.1:8001".to_string(),
            model: "qwen3-asr-0.6b".to_string(),
            client,
        }
    }

    /// Transcribe audio using external ASR API (blocking version)
    ///
    /// # Arguments
    /// * `wav_data` - WAV file data as bytes
    ///
    /// # Returns
    /// * `Result<String>` - Transcribed text
    pub fn transcribe_blocking(wav_data: &[u8]) -> Result<String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        let url = "http://127.0.0.1:8001/v1/audio/transcriptions";

        info!(
            "🎤 [External ASR] Sending {} bytes to {}",
            wav_data.len(),
            url
        );
        debug!("Sending audio to external ASR API: {}", url);

        // Create multipart form
        let wav_part = multipart::Part::bytes(wav_data.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let form = multipart::Form::new()
            .part("file", wav_part)
            .text("model", "qwen3-asr-0.6b");

        // Send request (blocking)
        let response = client
            .post(url)
            .multipart(form)
            .send()
            .context("Failed to send request to ASR API")?;

        // Check status
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().unwrap_or_default();
            error!("ASR API returned error status {}: {}", status, error_text);
            anyhow::bail!("ASR API error: {} - {}", status, error_text);
        }

        // Parse response
        #[derive(serde::Deserialize)]
        struct ASRResponse {
            text: String,
            #[allow(dead_code)]
            model: Option<String>,
        }

        let result: ASRResponse = response.json().context("Failed to parse ASR response")?;

        info!("ASR transcription completed: {} chars", result.text.len());

        Ok(result.text)
    }
}

impl Default for ExternalASRClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = ExternalASRClient::new();
        assert_eq!(client.base_url, "http://127.0.0.1:8001");
        assert_eq!(client.model, "qwen3-asr-0.6b");
    }
}
