use std::{
    io::{self, BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc,
    thread,
};
use anyhow::{Result, Context};

pub struct PythonRunner {
    child: Child,
    output_receiver: mpsc::Receiver<String>,
}

impl PythonRunner {
    pub fn new(script_path: PathBuf, args: Vec<String>) -> Result<Self> {
        // Verify file exists and has .py extension
        if !script_path.exists() {
            anyhow::bail!("Python script not found: {:?}", script_path);
        }
        if script_path.extension().and_then(|ext| ext.to_str()) != Some("py") {
            anyhow::bail!("File must have .py extension: {:?}", script_path);
        }

        // Start Python process
        let mut child = Command::new("python")
            .arg(script_path)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to start Python process")?;

        let stdout = child.stdout.take()
            .context("Failed to capture stdout")?;
        let stderr = child.stderr.take()
            .context("Failed to capture stderr")?;

        let (tx, rx) = mpsc::channel();

        // Handle stdout
        let stdout_tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if stdout_tx.send(line).is_err() {
                        break;
                    }
                }
            }
        });

        // Handle stderr
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if tx.send(format!("ERROR: {}", line)).is_err() {
                        break;
                    }
                }
            }
        });

        Ok(PythonRunner {
            child,
            output_receiver: rx,
        })
    }

    pub fn try_recv(&self) -> Result<Option<String>> {
        match self.output_receiver.try_recv() {
            Ok(line) => Ok(Some(line)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => 
                anyhow::bail!("Python process output channel disconnected"),
        }
    }

    pub fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }
}

impl Drop for PythonRunner {
    fn drop(&mut self) {
        let _ = self.kill();
    }
}
