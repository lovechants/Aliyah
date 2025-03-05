use std::thread;
use std::sync::mpsc;
use std::time::Duration;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
pub struct Update {
    #[serde(rename = "type")]
    pub type_: String,
    pub timestamp: f64,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct Command {
    pub command: String,
}

pub struct ZMQServer {
    context: zmq::Context,
    command_socket: zmq::Socket,  // REQ socket
    metrics_socket: zmq::Socket,  // SUB socket
}

fn log_to_file(msg: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/aliyah_debug.log")
    {
        let now = chrono::Local::now();
        let _ = writeln!(file, "[{}] {}", now.format("%Y-%m-%d %H:%M:%S.%3f"), msg);
    }
}

impl ZMQServer {

    pub fn new() -> Result<(Self, mpsc::Receiver<Update>, mpsc::Sender<String>)> {
        log_to_file("Initializing ZMQ server");
        
        let context = zmq::Context::new();
        
        // Setup command socket (REQ)
        log_to_file("Setting up command socket");
        let command_socket = context.socket(zmq::REQ)?;
        command_socket.set_linger(0)?;
        command_socket.set_rcvtimeo(1000)?;
        command_socket.set_sndtimeo(1000)?;
        command_socket.connect("tcp://127.0.0.1:5555")?;
        
        // Setup metrics socket (SUB)
        log_to_file("Setting up metrics socket");
        let metrics_socket = context.socket(zmq::SUB)?;
        metrics_socket.set_linger(0)?;
        metrics_socket.set_rcvtimeo(1000)?;
        metrics_socket.connect("tcp://127.0.0.1:5556")?;
        metrics_socket.set_subscribe(b"")?;
        
        let (update_tx, update_rx) = mpsc::channel();
        let (command_tx, command_rx) = mpsc::channel();
        
        let server = Self {
            context,
            command_socket,
            metrics_socket,
        };
        
        // Start listening for updates
        server.start_metrics_listener(update_tx.clone())?;
        
        // Start command handler
        server.start_command_handler(command_rx)?;
        
        log_to_file("ZMQ server initialized successfully");
        Ok((server, update_rx, command_tx))
    }
    /*
    fn start_metrics_listener(&self, tx: mpsc::Sender<Update>) -> Result<()> {
        let metrics_socket = self.context.socket(zmq::SUB)?;
        metrics_socket.connect("tcp://127.0.0.1:5556")?;
        metrics_socket.set_subscribe(b"")?;
        
        thread::spawn(move || {
            log_to_file("Metrics listener started");
            loop {
                match metrics_socket.recv_string(0) {
                    Ok(Ok(message)) => {
                        if let Ok(update) = serde_json::from_str::<Update>(&message) {
                            if tx.send(update).is_err() {
                                break;
                            }
                        }
                    }
                    Ok(Err(_)) => (),
                    Err(zmq::Error::EAGAIN) => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => {
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            }
            log_to_file("Metrics listener exiting");
        });
        
        Ok(())
    } */ 
    fn start_metrics_listener(&self, tx: mpsc::Sender<Update>) -> Result<()> {
        let metrics_socket = self.context.socket(zmq::SUB)?;
        metrics_socket.set_linger(0)?;
        metrics_socket.connect("tcp://127.0.0.1:5556")?;
        metrics_socket.set_subscribe(b"")?;
        
        thread::spawn(move || {
            log_to_file("Metrics listener started");
            loop {
                match metrics_socket.recv_string(0) {  // Changed from DONTWAIT to blocking mode
                    Ok(Ok(message)) => {
                        log_to_file(&format!("Received message: {}", message));
                        match serde_json::from_str::<Update>(&message) {
                            Ok(update) => {
                                log_to_file(&format!("Parsed update: {:?}", update));
                                if tx.send(update).is_err() {
                                    log_to_file("Channel closed, exiting listener");
                                    break;
                                }
                            },
                            Err(e) => log_to_file(&format!("Failed to parse update: {}", e))
                        }
                    }
                    Ok(Err(e)) => log_to_file(&format!("Invalid UTF-8: {:?}", e)),
                    Err(e) => {
                        if e == zmq::Error::EAGAIN {
                            thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        log_to_file(&format!("ZMQ error: {}", e));
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        });
        
        Ok(())
    }

    fn start_command_handler(&self, rx: mpsc::Receiver<String>) -> Result<()> {
        let command_socket = self.context.socket(zmq::REQ)?;
        command_socket.connect("tcp://127.0.0.1:5555")?;
        
        thread::spawn(move || {
            log_to_file("Command handler started");
            loop {
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(command) => {
                        let cmd = Command { command: command.clone() };
                        if let Ok(message) = serde_json::to_string(&cmd) {
                            if command_socket.send(&message, 0).is_err() {
                                log_to_file("Failed to send command");
                                break;
                            }
                            if command_socket.recv_msg(0).is_err() {
                                log_to_file("Failed to receive acknowledgment");
                                break;
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        log_to_file("Command channel disconnected");
                        break;
                    }
                }
            }
            log_to_file("Command handler exiting");
        });
        
        Ok(())
    }

    pub fn start_listening(&mut self, tx: mpsc::Sender<Update>) -> Result<()> {
        log_to_file("Starting listening for updates");
        self.start_metrics_listener(tx)?;
        Ok(())
    }

    pub fn send_command(&mut self, command: &str) -> Result<()> {
        log_to_file(&format!("Sending command: {}", command));
        let cmd = Command { command: command.to_string() };
        let message = serde_json::to_string(&cmd)?;
        
        // Create a new REQ socket for each command to avoid state issues
        let command_socket = self.context.socket(zmq::REQ)?;
        command_socket.set_linger(0)?;
        command_socket.set_rcvtimeo(1000)?;
        command_socket.set_sndtimeo(1000)?;
        command_socket.connect("tcp://127.0.0.1:5555")?;
        
        // Send the command
        if let Err(e) = command_socket.send(&message, 0) {
            log_to_file(&format!("Failed to send command: {}", e));
            return Err(anyhow::anyhow!("Failed to send command: {}", e));
        }
        
        // Wait for acknowledgment
        match command_socket.recv_string(0) {
            Ok(Ok(response)) => {
                if response == "ACK" {
                    log_to_file("Command acknowledged");
                    Ok(())
                } else {
                    log_to_file(&format!("Invalid acknowledgment: {}", response));
                    Err(anyhow::anyhow!("Invalid acknowledgment"))
                }
            }
            Ok(Err(e)) => {
                log_to_file(&format!("Invalid UTF-8 in response: {:?}", e));
                Err(anyhow::anyhow!("Invalid response encoding"))
            }
            Err(e) => {
                log_to_file(&format!("Failed to receive acknowledgment: {}", e));
                Err(anyhow::anyhow!("Failed to receive acknowledgment"))
            }
        }
    }
}

impl Drop for ZMQServer {
    fn drop(&mut self) {
        log_to_file("Cleaning up ZMQ server");
    }
}
