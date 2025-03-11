use std::fmt;

#[derive(Debug, Clone)]
pub enum ScriptState {
    Starting,
    Running, 
    Error(ScriptError),
    Completed,
    Stopped,
    Paused,
}

#[derive(Debug, Clone)]
pub enum ScriptError {
    ExectutionError(String),
    ParseError(String),
    RuntimeError(String),
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScriptError::ExectutionError(msg) => write!(f, "Exectution Error: {}", msg),
            ScriptError::ParseError(msg) => write!(f, "Parser Error: {}", msg),
            ScriptError::RuntimeError(msg) => write!(f, "Runtime Error: {}", msg),
        }
    }
}


