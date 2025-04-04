use std::path::Path;
use std::fs;
use anyhow::{Result, Context, bail};
use log::{debug, log, info, error};

// Can add some more frameworks 
// Next include Burn + SciKit Learn 
// Traditional ML techniques will probably require a new struct to define layers

#[derive(Debug, Clone)]
pub enum MLFramework {
    PyTorch,
    TensorFlow,
    JAX,
    Unknown,
    //Custom,
    Keras,
}

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub input_size: Option<Vec<usize>>,
    pub output_size: Option<Vec<usize>>,
    pub parameters: usize,
}

#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    pub framework: Option<MLFramework>,
    pub layers: Vec<LayerInfo>,
    pub total_parameters: usize,
}

pub fn parse_python_script(path: &Path) -> Result<ModelArchitecture> {
    info!("Starting to parse Python script: {:?}", path);
    
    let content = fs::read_to_string(path)
        .context("Failed to read Python script")?;
    
    debug!("Successfully read script content, length: {}", content.len());

    // Detect framework
    let framework = detect_framework(&content);
    info!("Detected framework: {:?}", framework);

    // Parse based on framework
    let result = match framework {
        MLFramework::PyTorch => parse_pytorch_model(&content),
        MLFramework::TensorFlow => parse_tensorflow_model(&content),
        MLFramework::JAX => parse_jax_model(&content),
        MLFramework::Unknown => {
            debug!("Unknown framework, attempting generic parsing");
            parse_generic_model(&content)
        }
        MLFramework::Keras => parse_keras_model(&content),
        //MLFramework::Custom => {
        //    debug!("Custom frame, work : TODO custom model config parsing");
        //    parse_from_custom(&content)
        //}
    };

    // Log the result
    match &result {
        Ok(arch) => info!("Successfully parsed model with {} layers", arch.layers.len()),
        Err(e) => error!("Failed to parse model: {}", e),
    }

    result
}

fn detect_framework(content: &str) -> MLFramework {
    if content.contains("import torch") || content.contains("from torch") {
        debug!("PyTorch imports found");
        MLFramework::PyTorch
    } else if content.contains("import tensorflow") || content.contains("from tensorflow") {
        debug!("TensorFlow imports found");
        MLFramework::TensorFlow
    } else if content.contains("import jax") || content.contains("from jax") {
        debug!("JAX imports found");
        MLFramework::JAX
    } else if content.contains("import keras") || content.contains("from keras") || content.contains("tensorflow.keras"){
        debug!("Keras imports found");
        MLFramework::Keras
    } else {
        debug!("No known ML framework imports found");
        MLFramework::Unknown
    }
}

fn parse_pytorch_model(content: &str) -> Result<ModelArchitecture> {
    debug!("Starting PyTorch model parsing");
    let mut layers = Vec::new();
    let mut total_parameters = 0;

    // Look for model definition
    let re = regex::Regex::new(r"(?m)^\s*(self\.|model\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(nn\.[a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)")
        .context("Failed to create regex pattern")?;
    
    debug!("Searching for layer definitions");
    for cap in re.captures_iter(content) {
        let layer_name = cap.get(2)
            .context("Failed to capture layer name")?
            .as_str()
            .to_string();
            
        let layer_type = cap.get(3)
            .context("Failed to capture layer type")?
            .as_str()
            .to_string();
            
        let params = cap.get(4)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();

        debug!("Found layer: {} of type {}", layer_name, layer_type);

        // Parse layer parameters
        let (input_size, output_size) = parse_layer_params(&params);
        debug!("Parsed sizes - input: {:?}, output: {:?}", input_size, output_size);
        
        // Estimate parameters
        let params_count = estimate_parameters(&layer_type, &input_size, &output_size, &params);
        total_parameters += params_count;

        layers.push(LayerInfo {
            name: layer_name,
            layer_type,
            input_size,
            output_size,
            parameters: params_count,
        });
    }

    if layers.is_empty() {
        bail!("No layers found in the model definition");
    }

    debug!("Successfully parsed {} layers", layers.len());
    Ok(ModelArchitecture {
        framework: Some(MLFramework::PyTorch),
        layers,
        total_parameters,
    })
}

fn parse_layer_params(params: &str) -> (Option<Vec<usize>>, Option<Vec<usize>>) {
    debug!("Parsing layer parameters: {}", params);
    let mut input_size = None;
    let mut output_size = None;

    // Parse common parameter patterns
    let parts: Vec<&str> = params.split(',').collect();
    for part in parts {
        let part = part.trim();
        if let Ok(size) = part.parse::<usize>() {
            if input_size.is_none() {
                debug!("Found input size: {}", size);
                input_size = Some(vec![size]);
            } else {
                debug!("Found output size: {}", size);
                output_size = Some(vec![size]);
            }
        }
    }

    (input_size, output_size)
}

fn estimate_parameters(layer_type: &str, input_size: &Option<Vec<usize>>, output_size: &Option<Vec<usize>>, layer_params: &str) -> usize {
    match layer_type {
        "nn.Linear" => {
            // Formula: (input_features * output_features) + output_features (bias)
            if let (Some(input), Some(output)) = (input_size, output_size) {
                if input.len() >= 1 && output.len() >= 1 {
                    let params = input[0] * output[0] + output[0];
                    debug!("Linear layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.Conv2d" => {
            // Formula: (kernel_height * kernel_width * in_channels * out_channels) + out_channels (bias)
            // The kernels used by default in Pytorch use the He initialisation from this paper: https://arxiv.org/abs/1502.01852
            if let (Some(input), Some(output)) = (input_size, output_size) {
                if input.len() >= 1 && output.len() >= 1 {
                    // Extract kernel size from params if available
                    let mut kernel_h = 3; 
                    let mut kernel_w = 3;
                    
                    // Try to extract kernel size from params - safely handle regex failures
                    if layer_params.contains("kernel_size") {
                        if let Ok(re) = regex::Regex::new(r"kernel_size=\(?(\d+)(?:\s*,\s*(\d+))?\)?") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(h_match) = caps.get(1) {
                                    if let Ok(h) = h_match.as_str().parse::<usize>() {
                                        kernel_h = h;
                                        if let Some(w_match) = caps.get(2) {
                                            if let Ok(w) = w_match.as_str().parse::<usize>() {
                                                kernel_w = w;
                                            } else {
                                                kernel_w = h; // Square kernel if w not parseable
                                            }
                                        } else {
                                            kernel_w = h; // Square kernel if w not captured
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    let in_channels = input[0];
                    let out_channels = output[0];
                    let params = (kernel_h * kernel_w * in_channels * out_channels) + out_channels;
                    debug!("Conv2d layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.BatchNorm2d" | "nn.BatchNorm1d" | "nn.BatchNorm3d" => {
            // Formula: 2 * num_features (gamma, beta) + 2 * num_features (running_mean, running_var)
            if let Some(features) = input_size.as_ref().or(output_size.as_ref()) {
                if !features.is_empty() {
                    let num_features = features[0];
                    let params = 4 * num_features; // 2 for gamma/beta + 2 for running mean/var
                    debug!("BatchNorm parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.ConvTranspose2d" => {
            // Formula: (kernel_height * kernel_width * in_channels * out_channels) + out_channels (bias)
            // Same as Conv2d but input/output channels are reversed
            if let (Some(input), Some(output)) = (input_size, output_size) {
                if input.len() >= 1 && output.len() >= 1 {
                    let mut kernel_h = 3;
                    let mut kernel_w = 3;
                    
                    if layer_params.contains("kernel_size") {
                        if let Ok(re) = regex::Regex::new(r"kernel_size=\(?(\d+)(?:\s*,\s*(\d+))?\)?") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(h_match) = caps.get(1) {
                                    if let Ok(h) = h_match.as_str().parse::<usize>() {
                                        kernel_h = h;
                                        if let Some(w_match) = caps.get(2) {
                                            if let Ok(w) = w_match.as_str().parse::<usize>() {
                                                kernel_w = w;
                                            } else {
                                                kernel_w = h;
                                            }
                                        } else {
                                            kernel_w = h;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    let in_channels = input[0];
                    let out_channels = output[0];
                    let params = (kernel_h * kernel_w * in_channels * out_channels) + out_channels;
                    debug!("ConvTranspose2d layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.Embedding" => {
            // Formula: num_embeddings * embedding_dim
            if let (Some(num_embeddings), Some(embedding_dim)) = (input_size, output_size) {
                if num_embeddings.len() >= 1 && embedding_dim.len() >= 1 {
                    let params = num_embeddings[0] * embedding_dim[0];
                    debug!("Embedding layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.LSTMCell" => {
            // Formula: 4 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size)
            if let (Some(input), Some(hidden)) = (input_size, output_size) {
                if input.len() >= 1 && hidden.len() >= 1 {
                    let input_size = input[0];
                    let hidden_size = hidden[0];
                    // 4 gates (input, forget, cell, output) each with input->hidden, hidden->hidden, and bias
                    let params = 4 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size);
                    debug!("LSTMCell layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.LSTM" => {
            // Formula for single-layer LSTM: 4 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size)
            // For multi-layer: Add (4 * ((hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size)) for each additional layer
            // Bidirectional doubles the total
            if let (Some(input), Some(hidden)) = (input_size, output_size) {
                if input.len() >= 1 && hidden.len() >= 1 {
                    let input_size = input[0];
                    let hidden_size = hidden[0];
                    
                    // Parse number of layers and bidirectional flag
                    let mut num_layers = 1;
                    let mut bidirectional = false;
                    
                    if layer_params.contains("num_layers") {
                        if let Ok(re) = regex::Regex::new(r"num_layers=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(l_match) = caps.get(1) {
                                    if let Ok(l) = l_match.as_str().parse::<usize>() {
                                        num_layers = l;
                                    }
                                }
                            }
                        }
                    }
                    
                    if layer_params.contains("bidirectional=True") {
                        bidirectional = true;
                    }
                    
                    // First layer
                    let mut params = 4 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size);
                    
                    // Additional layers
                    if num_layers > 1 {
                        params += (num_layers - 1) * 4 * ((hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size);
                    }
                    
                    // Bidirectional doubles the parameters
                    if bidirectional {
                        params *= 2;
                    }
                    
                    debug!("LSTM layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.GRU" => {
            // Formula for single-layer GRU: 3 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size)
            // For multi-layer: Add (3 * ((hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size)) for each additional layer
            // Bidirectional doubles the total
            if let (Some(input), Some(hidden)) = (input_size, output_size) {
                if input.len() >= 1 && hidden.len() >= 1 {
                    let input_size = input[0];
                    let hidden_size = hidden[0];
                    
                    // Parse number of layers and bidirectional flag
                    let mut num_layers = 1;
                    let mut bidirectional = false;
                    
                    if layer_params.contains("num_layers") {
                        if let Ok(re) = regex::Regex::new(r"num_layers=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(l_match) = caps.get(1) {
                                    if let Ok(l) = l_match.as_str().parse::<usize>() {
                                        num_layers = l;
                                    }
                                }
                            }
                        }
                    }
                    
                    if layer_params.contains("bidirectional=True") {
                        bidirectional = true;
                    }
                    
                    // First layer (3 gates: reset, update, new)
                    let mut params = 3 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size);
                    
                    // Additional layers
                    if num_layers > 1 {
                        params += (num_layers - 1) * 3 * ((hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size);
                    }
                    
                    // Bidirectional doubles the parameters
                    if bidirectional {
                        params *= 2;
                    }
                    
                    debug!("GRU layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.Transformer" => {
            // Will have to go through the docs for this one 
            // Much rather just do custom parsing for each "block" of the transformer 
            // Yea might have to seriously think about this one, just looking back at my GPT 
            // Maybe will test out a few transformer exampeles any from scratch transformer
            // Needs custom functions
            // Formula: d_model * (2 * d_model + 4 * d_ff + 8 * num_heads) * num_layers * 2
            if let Some(dim) = input_size.as_ref().or(output_size.as_ref()) {
                if !dim.is_empty() {
                    let d_model = dim[0];
                    
                    // Extract parameters
                    let mut num_layers = 6; // Default Transformer values
                    let mut num_heads = 8;
                    let mut d_ff = 4 * d_model; // Default feed-forward dimension
                    
                    // Try to parse parameters
                    if layer_params.contains("nhead") {
                        if let Ok(re) = regex::Regex::new(r"nhead=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(h_match) = caps.get(1) {
                                    if let Ok(h) = h_match.as_str().parse::<usize>() {
                                        num_heads = h;
                                    }
                                }
                            }
                        }
                    }
                    
                    if layer_params.contains("num_encoder_layers") {
                        if let Ok(re) = regex::Regex::new(r"num_encoder_layers=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(l_match) = caps.get(1) {
                                    if let Ok(l) = l_match.as_str().parse::<usize>() {
                                        num_layers = l;
                                    }
                                }
                            }
                        }
                    }
                    
                    if layer_params.contains("dim_feedforward") {
                        if let Ok(re) = regex::Regex::new(r"dim_feedforward=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(d_match) = caps.get(1) {
                                    if let Ok(d) = d_match.as_str().parse::<usize>() {
                                        d_ff = d;
                                    }
                                }
                            }
                        }
                    }
                    
                    let params = d_model * (2 * d_model + 4 * d_ff + 8 * num_heads) * num_layers * 2;
                    debug!("Transformer layer parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.LayerNorm" => {
            // Formula: 2 * normalized_shape (gamma and beta parameters)
            if let Some(shape) = input_size.as_ref().or(output_size.as_ref()) {
                if !shape.is_empty() {
                    let normalized_shape = shape[0];
                    let params = 2 * normalized_shape;
                    debug!("LayerNorm parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.GroupNorm" => {
            // Formula: 2 * num_channels (gamma and beta parameters)
            if let Some(shape) = input_size.as_ref().or(output_size.as_ref()) {
                if !shape.is_empty() {
                    let num_channels = shape[0];
                    let params = 2 * num_channels;
                    debug!("GroupNorm parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.TransformerEncoder" => {
            // Similar to transformer but only encoder part
            if let Some(dim) = input_size.as_ref().or(output_size.as_ref()) {
                if !dim.is_empty() {
                    let d_model = dim[0];
                    
                    // Extract parameters
                    let mut num_layers = 6; // Default
                    let mut num_heads = 8;
                    let mut d_ff = 4 * d_model; // Default
                    
                    if layer_params.contains("num_layers") {
                        if let Ok(re) = regex::Regex::new(r"num_layers=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(l_match) = caps.get(1) {
                                    if let Ok(l) = l_match.as_str().parse::<usize>() {
                                        num_layers = l;
                                    }
                                }
                            }
                        }
                    }
                    
                    if layer_params.contains("nhead") {
                        if let Ok(re) = regex::Regex::new(r"nhead=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(h_match) = caps.get(1) {
                                    if let Ok(h) = h_match.as_str().parse::<usize>() {
                                        num_heads = h;
                                    }
                                }
                            }
                        }
                    }
                    
                    if layer_params.contains("dim_feedforward") {
                        if let Ok(re) = regex::Regex::new(r"dim_feedforward=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(d_match) = caps.get(1) {
                                    if let Ok(d) = d_match.as_str().parse::<usize>() {
                                        d_ff = d;
                                    }
                                }
                            }
                        }
                    }
                    
                    let params = d_model * (2 * d_model + 4 * d_ff + 8 * num_heads) * num_layers;
                    debug!("TransformerEncoder parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.Conv1d" => {
            // Formula: (kernel_length * in_channels * out_channels) + out_channels (bias)
            if let (Some(input), Some(output)) = (input_size, output_size) {
                if input.len() >= 1 && output.len() >= 1 {
                    // Extract kernel size from params if available
                    let mut kernel_size = 3; // Default
                    
                    if layer_params.contains("kernel_size") {
                        if let Ok(re) = regex::Regex::new(r"kernel_size=(\d+)") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(k_match) = caps.get(1) {
                                    if let Ok(k) = k_match.as_str().parse::<usize>() {
                                        kernel_size = k;
                                    }
                                }
                            }
                        }
                    }
                    
                    let in_channels = input[0];
                    let out_channels = output[0];
                    let params = (kernel_size * in_channels * out_channels) + out_channels;
                    debug!("Conv1d parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.Conv3d" => {
            // Formula: (kernel_depth * kernel_height * kernel_width * in_channels * out_channels) + out_channels (bias)
            if let (Some(input), Some(output)) = (input_size, output_size) {
                if input.len() >= 1 && output.len() >= 1 {
                    // Default 3x3x3 kernel
                    let mut kernel_d = 3;
                    let mut kernel_h = 3;
                    let mut kernel_w = 3;
                    
                    if layer_params.contains("kernel_size") {
                        if let Ok(re) = regex::Regex::new(r"kernel_size=\(?(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\)?") {
                            if let Some(caps) = re.captures(layer_params) {
                                if let Some(d_match) = caps.get(1) {
                                    if let Ok(d) = d_match.as_str().parse::<usize>() {
                                        kernel_d = d;
                                        
                                        // If provided, get height or use depth as default
                                        if let Some(h_match) = caps.get(2) {
                                            if let Ok(h) = h_match.as_str().parse::<usize>() {
                                                kernel_h = h;
                                            } else {
                                                kernel_h = d;
                                            }
                                        } else {
                                            kernel_h = d;
                                        }
                                        
                                        // If provided, get width or use height as default
                                        if let Some(w_match) = caps.get(3) {
                                            if let Ok(w) = w_match.as_str().parse::<usize>() {
                                                kernel_w = w;
                                            } else {
                                                kernel_w = kernel_h;
                                            }
                                        } else {
                                            kernel_w = kernel_h;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    let in_channels = input[0];
                    let out_channels = output[0];
                    let params = (kernel_d * kernel_h * kernel_w * in_channels * out_channels) + out_channels;
                    debug!("Conv3d parameters: {}", params);
                    return params;
                }
            }
            0
        },

        "nn.PReLU" => {
            // Formula: If num_parameters=1, then 1 parameter; otherwise, one per channel
            if let Some(shape) = input_size.as_ref().or(output_size.as_ref()) {
                if !shape.is_empty() {
                    let num_channels = shape[0];
                    
                    let mut params = 1; // Default is 1 parameter
                    
                    if layer_params.contains("num_parameters") {
                        if layer_params.contains("num_parameters=1") {
                            params = 1;
                        } else {
                            params = num_channels;
                        }
                    }
                    
                    debug!("PReLU parameters: {}", params);
                    return params;
                }
            }
            0
        },

        // No trainable parameters in these layers
        "nn.ReLU" | "nn.Sigmoid" | "nn.Tanh" | "nn.MaxPool1d" | "nn.MaxPool2d" | 
        "nn.MaxPool3d" | "nn.AvgPool1d" | "nn.AvgPool2d" | "nn.AvgPool3d" | 
        "nn.AdaptiveAvgPool1d" | "nn.AdaptiveAvgPool2d" | "nn.AdaptiveAvgPool3d" |
        "nn.Flatten" | "nn.Dropout" | "nn.Dropout2d" | "nn.Dropout3d" => {
            debug!("{} has no trainable parameters", layer_type);
            0
        },

        _ => {
            // Default case for unknown layers
            debug!("Unknown layer type: {}", layer_type);
            0
        }
    }
}

// Placeholder implementations that at least don't panic
fn parse_tensorflow_model(_content: &str) -> Result<ModelArchitecture> {
    debug!("TensorFlow parsing not implemented");
    Ok(ModelArchitecture {
        framework: Some(MLFramework::TensorFlow),
        layers: Vec::new(),
        total_parameters: 0,
    })
}

fn parse_jax_model(_content: &str) -> Result<ModelArchitecture> {
    debug!("JAX parsing not implemented");
    Ok(ModelArchitecture {
        framework: Some(MLFramework::JAX),
        layers: Vec::new(),
        total_parameters: 0,
    })
}

fn parse_generic_model(_content: &str) -> Result<ModelArchitecture> {
    debug!("Generic model parsing not implemented");
    Ok(ModelArchitecture {
        framework: Some(MLFramework::Unknown),
        layers: Vec::new(),
        total_parameters: 0,
    })
}

fn parse_keras_model(_cotnent: &str) -> Result<ModelArchitecture> {
    debug!("Keras parsing not implemented");
    Ok(ModelArchitecture {
        framework: Some(MLFramework::Keras),
        layers: Vec::new(),
        total_parameters: 0, 
    })
}

fn parse_from_custom(_content: &str) -> Result<ModelArchitecture> {
    debug!("Custom Model parsing requires setup from config"); 
    Ok(ModelArchitecture {
        framework: Some(MLFramework::Keras),
        layers: Vec::new(),
        total_parameters: 0, 
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_detection() {
        let pytorch_code = "import torch\nfrom torch import nn";
        assert!(matches!(detect_framework(pytorch_code), MLFramework::PyTorch));

        let tensorflow_code = "import tensorflow as tf";
        assert!(matches!(detect_framework(tensorflow_code), MLFramework::TensorFlow));
    }

    #[test]
    fn test_pytorch_parsing() {
        let model_code = r#"
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
"#;
        let arch = parse_pytorch_model(model_code).unwrap();
        assert_eq!(arch.layers.len(), 2);
        assert_eq!(arch.layers[0].layer_type, "nn.Linear");
    }
}
