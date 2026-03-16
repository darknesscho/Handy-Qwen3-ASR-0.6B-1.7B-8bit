use log::{debug, error, info};
use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

static PYTHON_COMMAND_CACHE: OnceLock<String> = OnceLock::new();

/// Get the path to the embedded Python executable
fn get_embedded_python_path() -> Option<PathBuf> {
    // Try to find embedded Python in app resources
    // This works for both development and production builds

    // First, check if we're in a Tauri app context
    if let Ok(app_dir) = std::env::current_exe() {
        // For macOS app bundle: Handy.app/Contents/MacOS/handy
        // Resources are in: Handy.app/Contents/Resources/
        let resources_dir = app_dir
            .parent() // MacOS/
            .and_then(|p| p.parent()) // Contents/
            .map(|p| p.join("Resources"));

        if let Some(resources) = resources_dir {
            let embedded_python = resources.join("python/bin/python3");
            if embedded_python.exists() {
                return Some(embedded_python);
            }

            // Also check for python3.11 specifically
            let embedded_python311 = resources.join("python/bin/python3.11");
            if embedded_python311.exists() {
                return Some(embedded_python311);
            }
        }
    }

    // Check environment variable for custom Python path
    if let Ok(python_path) = std::env::var("HANDY_EMBEDDED_PYTHON") {
        let path = PathBuf::from(python_path);
        if path.exists() {
            return Some(path);
        }
    }

    // Fallback: check if embedded python exists in current working directory
    let local_python = PathBuf::from("src-tauri/resources/python/bin/python3.11");
    if local_python.exists() {
        return Some(local_python);
    }

    let local_python2 = PathBuf::from("src-tauri/resources/python/bin/python3");
    if local_python2.exists() {
        return Some(local_python2);
    }

    None
}

fn add_python_candidate(candidates: &mut Vec<String>, candidate: String) {
    if !candidate.is_empty() && !candidates.iter().any(|c| c == &candidate) {
        candidates.push(candidate);
    }
}

fn python_supports_qwen3(command: &str) -> Result<(), String> {
    let output = Command::new(command)
        .args([
            "-c",
            "import mlx; from mlx_audio.stt import load as load_stt; print('ok')",
        ])
        .output()
        .map_err(|e| format!("spawn failed: {}", e))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let reason = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit code {:?}", output.status.code())
    };

    Err(reason)
}

pub fn resolve_python_command() -> std::result::Result<String, Box<dyn std::error::Error>> {
    let mut candidates = Vec::new();

    // Priority 1: Check for uv virtual environment in current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let uv_venv_python = cwd.join(".venv/bin/python3");
        if uv_venv_python.exists() {
            info!("Found uv virtual environment Python: {}", uv_venv_python.display());
            add_python_candidate(&mut candidates, uv_venv_python.to_string_lossy().to_string());
        }
    }

    if let Some(embedded) = get_embedded_python_path() {
        add_python_candidate(&mut candidates, embedded.to_string_lossy().to_string());
    } else {
        info!("Embedded Python not found, trying system Python candidates");
    }

    if let Ok(python_path) = std::env::var("HANDY_EMBEDDED_PYTHON") {
        add_python_candidate(&mut candidates, python_path);
    }
    if let Ok(python_path) = std::env::var("HANDY_PYTHON") {
        add_python_candidate(&mut candidates, python_path);
    }

    for candidate in [
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        "python3",
    ] {
        add_python_candidate(&mut candidates, candidate.to_string());
    }

    let mut failure_reasons = Vec::new();

    for candidate in candidates {
        if candidate.contains('/') && !Path::new(&candidate).exists() {
            continue;
        }

        match python_supports_qwen3(&candidate) {
            Ok(()) => {
                info!("Using Python interpreter for Qwen3: {}", candidate);
                return Ok(candidate);
            }
            Err(reason) => {
                failure_reasons.push(format!("{} => {}", candidate, reason));
            }
        }
    }

    Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "No compatible Python found for Qwen3 (requires `mlx` and `mlx_audio.stt.load`). Attempts: {}",
            failure_reasons.join(" | ")
        ),
    )))
}

/// Get the Python command to use (embedded or system)
fn get_python_command() -> std::result::Result<(String, Vec<String>), Box<dyn std::error::Error>>
{
    if let Some(cached) = PYTHON_COMMAND_CACHE.get() {
        return Ok((cached.clone(), vec![]));
    }

    let python = resolve_python_command()?;
    let _ = PYTHON_COMMAND_CACHE.set(python.clone());
    Ok((python, vec![]))
}

/// Qwen3 ASR Engine using MLX framework (macOS only)
pub struct Qwen3Engine {
    model_path: Option<String>,
    child_process: Option<Arc<Mutex<Child>>>,
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    stdout: Option<Arc<Mutex<BufReader<ChildStdout>>>>,
}

/// Parameters for Qwen3 inference
#[derive(Debug, Clone, Serialize)]
pub struct Qwen3InferenceParams {
    pub language: Option<String>,
}

impl Default for Qwen3InferenceParams {
    fn default() -> Self {
        Self { language: None }
    }
}

/// Result from transcription
#[derive(Debug, Clone)]
pub struct Qwen3TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
}

impl Qwen3Engine {
    pub fn new() -> Self {
        Self {
            model_path: None,
            child_process: None,
            stdin: None,
            stdout: None,
        }
    }

    pub fn load_model(
        &mut self,
        mlx_model_name: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        info!("Loading Qwen3 model (mlx-audio managed): {}", mlx_model_name);

        // Check MLX is available
        self.check_mlx_available()?;

        // Check mlx-audio is available
        self.check_mlx_audio_available()?;

        // Start persistent Python server process
        self.start_server(mlx_model_name)?;

        // Qwen3 model is managed externally by mlx-audio
        self.model_path = Some(mlx_model_name.to_string());
        info!(
            "Qwen3 model loaded successfully (mlx-audio managed): {}",
            mlx_model_name
        );
        Ok(())
    }

    pub fn unload_model(&mut self) {
        debug!("Unloading Qwen3 model");
        self.model_path = None;
        // Kill the server process
        if let Some(child) = self.child_process.take() {
            if let Ok(mut child) = Arc::try_unwrap(child) {
                let _ = child.get_mut().unwrap().kill();
            }
        }
        self.stdin = None;
        self.stdout = None;
    }

    fn start_server(
        &mut self,
        mlx_model_name: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let script = include_str!("../../resources/scripts/qwen3_asr_server.py");

        info!(
            "Starting Qwen3 ASR server process for model: {}",
            mlx_model_name
        );
        let start_time = std::time::Instant::now();

        // Get Python command (embedded or system)
        let (python_cmd, python_args) = get_python_command()?;

        // Build command with all arguments
        let mut cmd = Command::new(&python_cmd);
        for arg in &python_args {
            cmd.arg(arg);
        }

        let mut child = cmd
            .env("HANDY_QWEN3_MODEL", mlx_model_name)
            .arg("-c")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                error!("Failed to start Qwen3 server with '{}': {}", python_cmd, e);
                Box::new(e) as Box<dyn std::error::Error>
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            error!("Failed to get stdin from child process");
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to get stdin",
            ))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            error!("Failed to get stdout from child process");
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to get stdout",
            ))
        })?;

        // Wait for "READY" signal from server
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        // Set a timeout for waiting (30 seconds for model loading)
        let timeout = std::time::Duration::from_secs(30);
        let start_wait = std::time::Instant::now();

        loop {
            if start_wait.elapsed() > timeout {
                let _ = child.kill();
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Timeout waiting for Qwen3 server to be ready",
                )));
            }

            match reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF - process might have crashed
                    let mut stderr = String::new();
                    if let Some(mut stderr_pipe) = child.stderr.take() {
                        use std::io::Read;
                        let _ = stderr_pipe.read_to_string(&mut stderr);
                    }
                    error!("Qwen3 server process ended unexpectedly. stderr: {}", stderr);
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Qwen3 server process ended unexpectedly: {}", stderr),
                    )));
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed == "READY" {
                        info!("Qwen3 ASR server is ready (startup took {:?})", start_time.elapsed());
                        break;
                    } else {
                        // Log any other output as info (likely from model loading)
                        info!("Qwen3 server: {}", trimmed);
                    }
                }
                Err(e) => {
                    let _ = child.kill();
                    return Err(Box::new(e));
                }
            }
            line.clear();
        }

        self.child_process = Some(Arc::new(Mutex::new(child)));
        self.stdin = Some(Arc::new(Mutex::new(stdin)));
        self.stdout = Some(Arc::new(Mutex::new(reader)));

        Ok(())
    }

    pub fn transcribe_samples<P: serde::Serialize>(
        &mut self,
        audio: Vec<f32>,
        params: Option<P>,
    ) -> std::result::Result<Qwen3TranscriptionResult, Box<dyn std::error::Error>> {
        let _model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Model not loaded",
            )))?;

        // Ensure server is running
        if self.stdin.is_none() || self.stdout.is_none() {
            let mlx_model_name = self.model_path.clone().ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Model not loaded",
                )) as Box<dyn std::error::Error>
            })?;
            self.start_server(&mlx_model_name)?;
        }

        let transcribe_start = std::time::Instant::now();
        debug!("Transcribing {} samples with Qwen3", audio.len());

        // Serialize params if provided
        let params_json = match params {
            Some(p) => serde_json::to_string(&p).unwrap_or_else(|_| "{}".to_string()),
            None => "{}".to_string(),
        };

        // Prepare input data - use compact JSON to reduce data size
        let input_data = serde_json::json!({
            "audio": audio,
            "params": params_json,
        });

        // Write request to server
        {
            let stdin = self.stdin.as_ref().ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Server stdin not available",
                ))
            })?;

            let mut stdin = stdin.lock().map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to lock stdin: {}", e),
                ))
            })?;

            let json_line = format!("{}\n", input_data.to_string());
            stdin.write_all(json_line.as_bytes()).map_err(|e| {
                error!("Failed to write to server stdin: {}", e);
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            stdin.flush().map_err(|e| {
                error!("Failed to flush stdin: {}", e);
                Box::new(e) as Box<dyn std::error::Error>
            })?;
        }

        // Read response from server
        let response_line = {
            let stdout = self.stdout.as_ref().ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Server stdout not available",
                ))
            })?;

            let mut stdout = stdout.lock().map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to lock stdout: {}", e),
                ))
            })?;

            let mut line = String::new();
            stdout.read_line(&mut line).map_err(|e| {
                error!("Failed to read from server stdout: {}", e);
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            line
        };

        let parse_start = std::time::Instant::now();
        let result: serde_json::Value = serde_json::from_str(&response_line)
            .map_err(|e| {
                error!("Failed to parse response: {}. Response: {}", e, response_line);
                Box::new(e) as Box<dyn std::error::Error>
            })?;
        debug!("JSON parsing took {:?}", parse_start.elapsed());

        if let Some(error) = result.get("error") {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Qwen3 error: {}", error),
            )));
        }

        let text = result
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();

        let language = result
            .get("language")
            .and_then(|l| l.as_str())
            .map(|s| s.to_string());

        info!("Qwen3 transcription completed in {:?}", transcribe_start.elapsed());

        Ok(Qwen3TranscriptionResult { text, language })
    }

    fn check_mlx_available(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let _ = get_python_command()?;
        Ok(())
    }

    fn check_mlx_audio_available(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let _ = get_python_command()?;
        Ok(())
    }
}

impl Default for Qwen3Engine {
    fn default() -> Self {
        Self::new()
    }
}

// Manual Clone implementation since Child doesn't implement Clone
impl Clone for Qwen3Engine {
    fn clone(&self) -> Self {
        // Create a new instance - the child process cannot be cloned
        // The new instance will need to start its own server if needed
        Self {
            model_path: self.model_path.clone(),
            child_process: None,
            stdin: None,
            stdout: None,
        }
    }
}
