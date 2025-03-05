use std::os::unix::net::{UnixListener, UnixStream};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::thread;
use std::sync::mpsc;
use std::time::Duration;
use std::fs::OpenOptions;

/*
#[derive(Debug, Deserialize)]
pub struct Update {
    pub type_: String,
    pub timestamp: f64,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct Command {
    pub command: String,
}
*/ 

pub struct IPCServer {
    socket_path: PathBuf,
    listener: UnixListener,
    stream: Option<UnixStream>,
}

fn log_to_file(msg: &str) {
    if let Ok(mut file) = OpenOptions::new() 
        .create(true)
        .append(true)
        .open("/tmp/aliyah_debug.log")
    {
        let now = chrono::Local::now(); 
        let _ = writeln!(file, "[{}] {}", now.format("%Y-%m-%d %H:%M:%S.%3f"), msg);
    }
}
/*
impl IPCServer {
    pub fn new() -> Result<(Self, mpsc::Receiver<Update>)> {
        log_to_file("Creating new IPC server");
        let socket_path = PathBuf::from("/tmp/aliyah.sock");
        let _ = std::fs::remove_file(&socket_path);
        
        let listener = UnixListener::bind(&socket_path)?;
        log_to_file("Socket bound successfully");
        
        let (tx, rx) = mpsc::channel();
        
        Ok((Self {
            socket_path,
            listener,
            stream: None,
        }, rx))
    }

    pub fn accept_connection(&mut self) -> Result<()> {
        log_to_file("Waiting for connection...");
        let (stream, _) = self.listener.accept()?;
        log_to_file("Connection accepted");
        self.stream = Some(stream);
        Ok(())
    }

    pub fn send_command(&mut self, command: &str) -> Result<()> {
        log_to_file(&format!("Attempting to send command: {}", command));
        if let Some(stream) = &mut self.stream {
            let cmd = Command { command: command.to_string() };
            let message = serde_json::to_string(&cmd)?;
            writeln!(stream, "{}", message)?;
            stream.flush()?;
            log_to_file("Command sent successfully");
            Ok(())
        } else {
            log_to_file("Failed to send command: no active stream");
            Err(anyhow::anyhow!("No active stream"))
        }
    }

    pub fn start_listening(&mut self, tx: mpsc::Sender<Update>) -> Result<()> {
        log_to_file("Starting listener thread");
        if let Some(stream) = &self.stream {
            let reader = BufReader::new(stream.try_clone()?);
            
            thread::spawn(move || {
                log_to_file("Listener thread started");
                for line in reader.lines() {
                    match line {
                        Ok(line) => {
                            log_to_file(&format!("Received: {}", line));
                            if let Ok(update) = serde_json::from_str::<Update>(&line) {
                                if tx.send(update).is_err() {
                                    log_to_file("Failed to send update through channel");
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            log_to_file(&format!("Error reading line: {}", e));
                            break;
                        }
                    }
                }
                log_to_file("Listener thread exiting");
            });
        }
        Ok(())
    }
    pub fn start_command_handler(&mut self, rx: mpsc::Receiver<String>) -> Result<()> {
        if let Some(stream) = &self.stream {
            let mut stream = stream.try_clone()?;
            thread::spawn(move || {
                for command in rx {
                    log_to_file(&format!("Processing command: {}", command));
                    let cmd = Command { command: command.clone() };
                    match serde_json::to_string(&cmd) {
                        Ok(message) => {
                            if let Err(e) = writeln!(stream, "{}", message) {
                                log_to_file(&format!("Failed to send command: {}", e));
                                break;
                            }
                            if let Err(e) = stream.flush() {
                                log_to_file(&format!("Failed to flush stream: {}", e));
                                break;
                            }
                            log_to_file("Command sent successfully");
                        }
                        Err(e) => {
                            log_to_file(&format!("Failed to serialize command: {}", e));
                            break;
                        }
                    }
                }
                log_to_file("Command handler thread exiting");
            });
        }
        Ok(())
    }

}

impl Drop for IPCServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

*/ 

#[derive(Debug, Deserialize)]
pub struct Update {
    pub type_: String,
    pub timestamp: f64,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct Command {
    pub command: String,
}

pub struct CommandServer {
    socket_path: PathBuf,
    stream: Option<UnixStream>
}

pub struct UpdateServer {
    socket_path: PathBuf,
    stream: Option<UnixStream>,
}

impl CommandServer {
    pub fn new() -> Result<Self> {
        let socket_path = PathBuf::from("/tmp/aliyah_cmd.sock");
        let _ = std::fs::remove_file(&socket_path);
        
        let listener = UnixListener::bind(&socket_path)?;
        let (stream, _) = listener.accept()?;
        
        Ok(Self {
            socket_path,
            stream: Some(stream)
        })
    }

    pub fn send_command(&mut self, cmd: &str) -> Result<()> {
        if let Some(stream) = &mut self.stream {
            let cmd = Command { command: cmd.to_string() };
            writeln!(stream, "{}", serde_json::to_string(&cmd)?)?;
            stream.flush()?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("No command connection"))
        }
    }
}

impl UpdateServer {
    pub fn new() -> Result<(Self, mpsc::Receiver<Update>)> {
        let socket_path = PathBuf::from("/tmp/aliyah_update.sock");
        let _ = std::fs::remove_file(&socket_path);
        
        let listener = UnixListener::bind(&socket_path)?;
        let (tx, rx) = mpsc::channel();
        
        let (stream, _) = listener.accept()?;
        
        let server = Self {
            socket_path,
            stream: Some(stream),
        };

        // Start update listener thread
        if let Some(stream) = &server.stream {
            let reader = BufReader::new(stream.try_clone()?);
            let tx = tx.clone();
            
            thread::spawn(move || {
                for line in reader.lines() {
                    match line {
                        Ok(line) => {
                            if let Ok(update) = serde_json::from_str::<Update>(&line) {
                                if tx.send(update).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(_) => break
                    }
                }
            });
        }

        Ok((server, rx))
    }
}

impl Drop for CommandServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

impl Drop for UpdateServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}
