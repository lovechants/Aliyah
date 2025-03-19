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
        let params_count = estimate_parameters(&layer_type, &input_size, &output_size);
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

fn estimate_parameters(layer_type: &str, input_size: &Option<Vec<usize>>, output_size: &Option<Vec<usize>>) -> usize {
    match (layer_type, input_size, output_size) {
        ("nn.Linear", Some(input), Some(output)) => {
            let params = input[0] * output[0] + output[0];
            debug!("Linear layer parameters: {}", params);
            params
        },
        ("nn.Conv2d", Some(input), Some(output)) => {
            if input.len() >= 2 && output.len() >= 2 {
                let params = input[0] * input[1] * output[0] * output[1] + output[0];
                debug!("Conv2d layer parameters: {}", params);
                params
            } else {
                debug!("Invalid Conv2d dimensions");
                0
            }
        },
        _ => {
            debug!("Unknown layer type or missing dimensions: {}", layer_type);
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
