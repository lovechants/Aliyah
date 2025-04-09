use log::{debug, error, log_enabled, info, Level};
use env_logger;
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect, Margin},
    style::{Color, Style, Modifier},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, GraphType, Dataset, canvas::{Canvas, Context, Line as CanvasLine, Points}},
    widgets::{Chart, Axis, Wrap},
    Terminal, Frame,
    symbols::Marker,
    prelude::Alignment,
};
use std::{
    io::{self, BufRead, BufReader, Write},
    process::{Child, Command as StdCommand, Stdio},
    thread,
    sync::mpsc,
    collections::HashMap,
    time::{Duration, Instant},
    path::PathBuf,
    error::Error,
};
use std::cmp::min;
use std::fs::OpenOptions;
use std::fmt;
use sysinfo::{CpuExt, System, SystemExt};
use clap::Parser;
use aliyah::{ PythonRunner, MLFramework, ModelArchitecture};
use aliyah::{ ScriptState, ScriptError };
use aliyah::ScriptOutput;
use aliyah::Update;
use aliyah::ZMQServer;
use aliyah::Command;


const DEBUG_LOGGING: bool = false;  
fn log_to_file(msg: &str) {
    if !DEBUG_LOGGING {
        return;  // Skip logging entirely when not debugging
    }
    
    // Use a static mutex to avoid multiple file opens
    static mut LOG_FILE: Option<std::fs::File> = None;
    static LOG_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    
    // Get lock to ensure thread safety
    let _guard = LOG_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    
    unsafe {
        // Initialize file only once
        if LOG_FILE.is_none() {
            let mut path = std::env::temp_dir();
            path.push("aliyah_debug.log");
            LOG_FILE = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok();
        }
        
        if let Some(file) = &mut LOG_FILE {
            let now = chrono::Local::now();
            let _ = writeln!(file, "[{}] {}", now.format("%Y-%m-%d %H:%M:%S.%3f"), msg);
            let _ = file.flush();  // Ensure it's written
        }
    }
}



#[derive(Parser)]
struct Cli {
    #[arg(name = "SCRIPT")]
    script: PathBuf,

    #[arg(last = true)]
    script_args: Vec<String>,

    #[arg(short, long)]
    debug: bool,
}

#[derive(Debug, Clone)]
struct SystemMetrics {
    cpu_usage: f32,
    memory_used: u64,
    memory_total: u64,
    gpu_info: Option<GpuInfo>,
    timestamp: Instant,
}

#[derive(Debug, Clone)]
struct GpuInfo {
    utilization: f32,
    memory_used: u64,
    memory_total: u64,
}


#[derive(Debug, Clone)]
struct TrainingMetrics {
    epoch: usize,
    metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct ModelPrediction {
    epoch: usize,
    timestamp: Instant,
    values: Vec<f64>,
    labels: Option<Vec<String>>,
    description: String, // Provide context for the output will be dependent on user
}


struct App {
    output_lines: Vec<String>,
    metrics_history: Vec<TrainingMetrics>,
    current_metrics: HashMap<String, serde_json::Value>,
    system_metrics: Option<SystemMetrics>,
    sys: System,
    network: NetworkLayout,
    last_viz_update: Instant,
    model_architecture: ModelArchitecture,
    script_state: ScriptState,
    error_log: Vec<String>,
    error_scroll: usize,
    show_error_logs: bool,
    is_paused: bool,
    training_scroll: usize,
    command_tx: Option<mpsc::Sender<String>>,
    zmq_server: Option<ZMQServer>,
    start_time: Option<Instant>,
    total_epochs: Option<usize>,
    total_batches: Option<usize>,
    current_epoch: Option<usize>,
    current_batch: Option<usize>,
    selected_node: Option<usize>,
    hover_position: Option<(f64, f64)>,
    show_model_output: bool,
    model_prediction: Option<ModelPrediction>,
    final_elapsed: Option<Duration>,
    paused_elapsed: Option<Duration>,
    hovered_node_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug)]
pub struct NetworkLayout {
    nodes: Vec<NetworkNode>,
    connections: Vec<NetworkConnection>,
    layers: Vec<usize>, // number of nodes in each layer
    bounds: (f64, f64, f64, f64), // (min_x, min_y, max_x, max_y)
}

pub enum IPCState {
    Connected,
    Disconnected,
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerCategory {
    Input,
    Output,
    Convolutional,
    Pooling,
    Normalization,
    Activation,
    Recurrent,
    Attention,
    Linear,
    Embedding,
    Dropout,
    Other(String),  
}

#[derive(Debug, Clone)]
pub struct NodeDetails {
    pub operation: String,       // Specific operation (ReLU, Softmax, etc)
    pub input_shape: Option<Vec<usize>>,
    pub output_shape: Option<Vec<usize>>,
    pub params_count: usize,
    pub activation_history: Vec<f64>, // Store recent activations for mini-histograms
}

#[derive(Debug, Clone)]
pub struct NetworkNode {
    id: usize,
    x: f64,
    y: f64,
    layer_index: usize,
    original_index: usize,
    scaled_index: usize,
    activation: Option<f64>,
    node_type: NodeType,
    //category: LayerCategory,
    //details: Option<NodeDetails>,
}

#[derive(Debug, Clone)]
pub struct NetworkConnection {
    from_node_id: usize,
    to_node_id: usize,
    weight: f64,
    active: bool,
    gradient: Option<f64>,  // For showing backprop
    signal_strength: Option<f64>,  // For showing forward signal strength
}

impl NetworkLayout {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut nodes = Vec::new();
        let mut connections = Vec::new();
        let mut next_node_id = 0;

        let total_layers = layer_sizes.len();
        let max_visible_nodes = 10;
        if layer_sizes.is_empty() {
            return NetworkLayout {
                nodes,
                connections,
                layers: Vec::new(),
                bounds: (-1.0, -1.0, 1.0, 1.0),
            };
        }
        // Calculate scaled sizes while maintaining proportions
        let scaled_sizes: Vec<usize> = layer_sizes.iter()
            .map(|&size| size.min(max_visible_nodes))
            .collect();

        // Calculate scale factors for showing which nodes represent multiple nodes
        let scale_factors: Vec<f64> = layer_sizes.iter()
            .zip(scaled_sizes.iter())
            .map(|(&orig, &scaled)| if scaled < orig {
                orig as f64 / scaled as f64
            } else {
                1.0
            })
            .collect();

        // Position nodes in a clean layout
        for (layer_idx, &size) in layer_sizes.iter().enumerate() {
            let x = -0.8 + (1.6 * layer_idx as f64 / (total_layers - 1) as f64);
            let scaled_size = scaled_sizes[layer_idx];

            for node_idx in 0..scaled_size {
                let y = if scaled_size > 1 {
                    -0.8 + (1.6 * node_idx as f64 / (scaled_size - 1) as f64)
                } else {
                    0.0
                };

                // Map scaled index to original index for accurate data representation
                let original_index = if size > max_visible_nodes {
                    ((node_idx as f64 * scale_factors[layer_idx]).round() as usize).min(size - 1)
                } else {
                    node_idx
                };

                nodes.push(NetworkNode {
                    id: next_node_id,
                    x,
                    y,
                    layer_index: layer_idx,
                    original_index,
                    scaled_index: node_idx,
                    activation: None,
                    node_type: if layer_idx == 0 {
                        NodeType::Input
                    } else if layer_idx == total_layers - 1 {
                        NodeType::Output
                    } else {
                        NodeType::Hidden
                    },
                });
                next_node_id += 1;
            }
        }

        // Create connections between layers showing information flow
        for layer_idx in 0..total_layers - 1 {
            let current_layer: Vec<_> = nodes.iter()
                .filter(|n| n.layer_index == layer_idx)
                .collect();
            let next_layer: Vec<_> = nodes.iter()
                .filter(|n| n.layer_index == layer_idx + 1)
                .collect();

            for &from_node in &current_layer {
                for &to_node in &next_layer {
                    connections.push(NetworkConnection {
                        from_node_id: from_node.id,
                        to_node_id: to_node.id,
                        weight: 1.0,
                        active: false,
                        gradient: None,
                        signal_strength: None,
                    });
                }
            }
        }

        NetworkLayout {
            nodes,
            connections,
            layers: layer_sizes.to_vec(),
            bounds: (-1.0, -1.0, 1.0, 1.0),
        }
    }

    // Handle updates from training
    pub fn update_forward_signal(&mut self, from_layer: usize, from_idx: usize,
                               to_layer: usize, to_idx: usize, signal: f64) {
        if self.nodes.is_empty() || self.connections.is_empty() {
            return;
        }
        
        // First try to find exact nodes
        let from_node = self.nodes.iter()
            .find(|n| n.layer_index == from_layer && n.original_index == from_idx);
            
        let to_node = self.nodes.iter()
            .find(|n| n.layer_index == to_layer && n.original_index == to_idx);
            
        // If we found both nodes, find the connection
        if let (Some(from), Some(to)) = (from_node, to_node) {
            if let Some(conn) = self.connections.iter_mut()
                .find(|c| c.from_node_id == from.id && c.to_node_id == to.id) {
                    
                conn.signal_strength = Some(signal);
                conn.active = signal > 0.01; // Lower threshold to make more connections visible
                debug!("Updated connection directly: from=({},{}), to=({},{}), signal={:.4}, active={}",
                       from_layer, from_idx, to_layer, to_idx, signal, conn.active);
                return;
            }
        }
        
        // If we couldn't find exact match, try approximate matching
        let from_nodes: Vec<_> = self.nodes.iter()
            .filter(|n| n.layer_index == from_layer)
            .collect();
            
        let to_nodes: Vec<_> = self.nodes.iter()
            .filter(|n| n.layer_index == to_layer)
            .collect();
            
        if !from_nodes.is_empty() && !to_nodes.is_empty() {
            // Find closest nodes by original index
            let closest_from = from_nodes.iter()
                .min_by_key(|n| ((n.original_index as i64) - (from_idx as i64)).abs() as usize)
                .unwrap();
                
            let closest_to = to_nodes.iter()
                .min_by_key(|n| ((n.original_index as i64) - (to_idx as i64)).abs() as usize)
                .unwrap();
                
            // Find and update the connection
            if let Some(conn) = self.connections.iter_mut()
                .find(|c| c.from_node_id == closest_from.id && c.to_node_id == closest_to.id) {
                    
                conn.signal_strength = Some(signal);
                conn.active = signal > 0.01;
                debug!("Updated approximate connection: want=({},{})->({},{}), using=({},{})->({},{}), signal={:.4}", 
                      from_layer, from_idx, to_layer, to_idx,
                      closest_from.layer_index, closest_from.original_index, 
                      closest_to.layer_index, closest_to.original_index,
                      signal);
            }
        }
    }

    pub fn update_backward_signal(&mut self, from_layer: usize, from_idx: usize,
                                to_layer: usize, to_idx: usize, gradient: f64) {
        if let Some(conn) = self.find_connection(from_layer, from_idx, to_layer, to_idx) {
            conn.gradient = Some(gradient);
        }
    }

    pub fn update_node_activation(&mut self, layer: usize, node: usize, activation: f64) {
        if let Some(node) = self.nodes.iter_mut()
            .find(|n| n.layer_index == layer && n.original_index == node) {
            node.activation = Some(activation);
        }
    }

    fn find_connection(&mut self, from_layer: usize, from_idx: usize,
                      to_layer: usize, to_idx: usize) -> Option<&mut NetworkConnection> {
        if self.nodes.is_empty() {
            return None;
        }
        let from_node = self.nodes.iter()
            .find(|n| n.layer_index == from_layer && n.original_index == from_idx)?;
        let to_node = self.nodes.iter()
            .find(|n| n.layer_index == to_layer && n.original_index == to_idx)?;

        self.connections.iter_mut()
            .find(|c| c.from_node_id == from_node.id && c.to_node_id == to_node.id)
    }

    pub fn draw<'a>(&'a self) -> Canvas<'a, impl Fn(&mut ratatui::widgets::canvas::Context<'_>) + 'a> {
        Canvas::default()
            .paint(|ctx| {
                // Draw connections with weight visualization
                for conn in &self.connections {
                    if let (Some(from), Some(to)) = (
                        self.nodes.iter().find(|n| n.id == conn.from_node_id),
                        self.nodes.iter().find(|n| n.id == conn.to_node_id)
                    ) {
                        // Determine line thickness and color based on weight and activity
                        let weight_abs = conn.weight.abs();
                        let weight_intensity = ((weight_abs * 200.0) as u8).min(255);
                        
                        let color = if conn.active {
                            if conn.weight > 0.0 {
                                Color::Rgb(0, weight_intensity, weight_intensity) // Positive weights in cyan
                            } else {
                                Color::Rgb(weight_intensity, 0, 0) // Negative weights in red
                            }
                        } else {
                            Color::DarkGray
                        };

                        // Draw connection line
                        ctx.draw(&CanvasLine {
                            x1: from.x,
                            y1: from.y,
                            x2: to.x,
                            y2: to.y,
                            color,
                        });

                        // Add signal flow animation if active
                        if conn.active {
                            let t = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as f64 / 1000.0;
                            
                            // Animate 3 particles along the connection
                            for i in 0..3 {
                                let phase = ((t * 2.0) + (i as f64 * 0.33)) % 1.0;
                                let x = from.x + (to.x - from.x) * phase;
                                let y = from.y + (to.y - from.y) * phase;
                                
                                ctx.draw(&Points {
                                    coords: &[(x, y)],
                                    color: if conn.weight > 0.0 {
                                        Color::Cyan
                                    } else {
                                        Color::Red
                                    },
                                });
                            }
                        }
                    }
                }

                // Draw nodes with more detailed visualization
                for node in &self.nodes {
                    // Determine node size based on type and activation
                    let base_radius = match node.node_type {
                        NodeType::Input => 0.07,
                        NodeType::Hidden => 0.06,
                        NodeType::Output => 0.07,
                    };
                    
                    // Scale radius slightly based on activation if available
                    let radius = if let Some(act) = node.activation {
                        base_radius * (1.0 + act.abs() * 0.5)
                    } else {
                        base_radius
                    };
                    
                    // Generate points for circle
                    let points = self.generate_circle_points(node.x, node.y, radius, 16);
                    
                    // Determine node color based on type and activation
                    let color = match node.node_type {
                        NodeType::Input => {
                            if let Some(act) = node.activation {
                                let intensity = ((act * 200.0) as u8).min(255).max(100);
                                Color::Rgb(100, 100, intensity) // Blue with activation intensity
                            } else {
                                Color::Blue
                            }
                        },
                        NodeType::Hidden => {
                            if let Some(act) = node.activation {
                                if act > 0.0 {
                                    let intensity = ((act * 200.0) as u8).min(255).max(50);
                                    Color::Rgb(intensity, intensity, intensity) // White with activation intensity
                                } else {
                                    Color::DarkGray
                                }
                            } else {
                                Color::DarkGray
                            }
                        },
                        NodeType::Output => {
                            if let Some(act) = node.activation {
                                let intensity = ((act * 200.0) as u8).min(255).max(100);
                                Color::Rgb(0, intensity, 0) // Green with activation intensity
                            } else {
                                Color::Green
                            }
                        }
                    };

                    // Draw node
                    ctx.draw(&Points {
                        coords: &points,
                        color,
                    });
                    
                    // Draw outline for activated nodes
                    if let Some(act) = node.activation {
                        if act.abs() > 0.3 {
                            let outline = self.generate_circle_points(node.x, node.y, radius * 1.1, 20);
                            ctx.draw(&Points {
                                coords: &outline,
                                color: Color::White,
                            });
                        }
                    }
                }
            })
            .x_bounds([self.bounds.0, self.bounds.2])
            .y_bounds([self.bounds.1, self.bounds.3])
    }

    fn generate_circle_points(&self, x: f64, y: f64, radius: f64, points: usize) -> Vec<(f64, f64)> {
        (0..points).map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64 / points as f64);
            (
                x + radius * angle.cos(),
                y + radius * angle.sin()
            )
        }).collect()
    }

    fn get_node_index(&self, pos: (usize, usize)) -> usize {
        let mut index = 0;
        for i in 0..pos.0 {
            index += self.layers[i];
        }
        index + pos.1
    }

    pub fn update_activation(&mut self, layer: usize, node: usize, activation: f64) {
        if self.nodes.is_empty() || layer >= self.layers.len() {
            return; 
        }
        if self.nodes.is_empty() {
            return;
        }
        let idx = self.get_node_index((layer, node));
        if let Some(node) = self.nodes.get_mut(idx) {
            node.activation = Some(activation);
        }
    }
    pub fn update_connection(&mut self, from: (usize, usize), to: (usize, usize), weight: f64, active: bool) {
        if self.nodes.is_empty() {
            return;
        }
        // First find both nodes safely
        let from_node_id = match self.nodes.iter()
            .find(|n| n.layer_index == from.0 && n.original_index == from.1)
            .map(|n| n.id) {
                Some(id) => id,
                None => return, // Early return if from_node not found
        };

        let to_node_id = match self.nodes.iter()
            .find(|n| n.layer_index == to.0 && n.original_index == to.1)
            .map(|n| n.id) {
                Some(id) => id,
                None => return, // Early return if to_node not found
        };

        // Now find and update the connection
        if let Some(conn) = self.connections.iter_mut()
            .find(|c| c.from_node_id == from_node_id && c.to_node_id == to_node_id) {
            conn.weight = weight;
            conn.active = active;
        }
    }

    pub fn find_nearest_node(&self, x: f64, y: f64) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        
        // Find closest node with improved selection radius
        let mut closest_idx = 0;
        let mut closest_distance = f64::MAX;
        
        for (idx, node) in self.nodes.iter().enumerate() {
            let distance = ((node.x - x).powi(2) + (node.y - y).powi(2)).sqrt();
            
            // Increase selection radius for smaller nodes to make them easier to select
            let selection_radius = match node.node_type {
                NodeType::Input => 0.15,   // Larger selection area for input nodes
                NodeType::Hidden => 0.12,  // Medium selection area for hidden nodes
                NodeType::Output => 0.15,  // Larger selection area for output nodes
            };
            
            if distance < selection_radius && distance < closest_distance {
                closest_distance = distance;
                closest_idx = idx;
            }
        }
        
        // Return closest node if it's within selection range
        if closest_distance < f64::MAX {
            Some(closest_idx)
        } else {
            None
        }
    }
    
    /*
    // Add visual hover effect to improve selection feedback
    pub fn draw_hover_indicator<'a>(&'a self, hover_position: Option<(f64, f64)>) -> impl Fn(&mut ratatui::widgets::canvas::Context<'_>) + 'a {
        move |ctx| {
            // First draw the normal network
            self.draw(None);
            
            // Then add hover indicator if applicable
            if let Some((x, y)) = hover_position {
                // Find the closest node to show hover effect
                if let Some(closest_idx) = self.find_nearest_node(x, y) {
                    let node = &self.nodes[closest_idx];
                    
                    // Draw hover highlight
                    let radius = 0.15; // Larger than the node for visibility
                    let hover_points = self.generate_circle_points(node.x, node.y, radius, 24);
                    
                    ctx.draw(&Points {
                        coords: &hover_points,
                        color: Color::Yellow, // Distinct color for hover state
                    });
                }
            }
        }
    }
    */
}

impl App {
    fn new() -> App {
        App {
            output_lines: Vec::new(),
            metrics_history: Vec::new(),
            current_metrics: HashMap::new(),
            system_metrics: None,
            sys: System::new_all(),
            network: NetworkLayout::new(&[]),
            model_architecture: ModelArchitecture { framework: None, layers: Vec::new(), total_parameters: 0},
            script_state: ScriptState::Starting,
            error_log: Vec::new(),
            error_scroll: 0,
            show_error_logs: false,
            is_paused: false,
            training_scroll: 0,
            command_tx: None,
            zmq_server: None,
            start_time: None,
            total_epochs: None,
            current_epoch: None,
            total_batches: None,
            current_batch: None,
            last_viz_update: Instant::now(),
            selected_node: None,
            hover_position: None,
            hovered_node_idx: None,
            model_prediction: None,
            show_model_output: false,
            final_elapsed: None,
            paused_elapsed: None,
        }
    }
    fn log_recieved_update(&self, update: &Update) {
        log_to_file(&format!(
                "Processing Update - Type: {}, Time: {}, Data: {:?}",
                update.type_,
                update.timestamp,
                update.data
        ));
    }

    fn toggle_output_view(&mut self) {
        self.show_model_output = !self.show_model_output;
    }

    fn terminal_to_canvas_coords(&self, col: u16, row: u16, term_width: u16, term_height: u16) -> (f64, f64) {
        // Terminal characters are typically ~2x taller than wide
        // Adjust aspect ratio to compensate for non-square character cells
        let aspect_correction = 0.5; 
        
        // Map to normalized -1.0 to 1.0 coordinate space with aspect correction
        let x = (col as f64 / term_width as f64) * 2.0 - 1.0;
        let y = ((row as f64 / term_height as f64) * 2.0 - 1.0) * aspect_correction;
        
        // Apply any scaling or offset from your canvas bounds
        let (min_x, min_y, max_x, max_y) = self.network.bounds;
        let canvas_width = max_x - min_x;
        let canvas_height = max_y - min_y;
        
        let canvas_x = min_x + (x + 1.0) * 0.5 * canvas_width;
        let canvas_y = min_y + (y + 1.0) * 0.5 * canvas_height;
        
        (canvas_x, canvas_y)
    }

    fn set_model_prediction(&mut self, values: Vec<f64>, labels: Option<Vec<String>>, description: String){
        self.model_prediction = Some(ModelPrediction {
            epoch: self.current_epoch.unwrap_or(0),
            timestamp: Instant::now(),
            values,
            labels,
            description,
        });

        self.output_lines.push("Model prediction captured".to_string());
    }

    fn handle_mouse_move(&mut self, col: u16, row: u16, term_width: u16, term_height: u16) {
        let (x, y) = self.terminal_to_canvas_coords(col, row, term_width, term_height);
        self.hover_position = Some((x, y));
        self.hovered_node_idx = None; // Reset hover state

        // Early exit if no network to check
        if self.network.nodes.is_empty() || self.network.layers.is_empty() {
            return;
        }

        let total_layers = self.network.layers.len();
        if total_layers == 0 { return; }

        
        let mut closest_layer_idx = 0;
        let mut min_dist_x = f64::MAX;

        for layer_idx in 0..total_layers {
             let layer_x = if total_layers > 1 {
                 -0.8 + (1.6 * layer_idx as f64 / (total_layers - 1) as f64)
             } else {
                 0.0 // Center if only one layer
             };
             let dist_x = (layer_x - x).abs();
             if dist_x < min_dist_x {
                 min_dist_x = dist_x;
                 closest_layer_idx = layer_idx;
             }
        }

        let mut layers_to_check = vec![closest_layer_idx];
        if closest_layer_idx > 0 {
            layers_to_check.push(closest_layer_idx - 1);
        }
        if closest_layer_idx + 1 < total_layers {
            layers_to_check.push(closest_layer_idx + 1);
        }

        let selection_radius: f64 = 0.15; // Keep the larger hit radius
        let selection_radius_sq = selection_radius.powi(2); // Pre-calculate squared radius
        let mut closest_node_dist_sq = f64::MAX;

        // Iterate only over nodes in the relevant layers
        for (idx, node) in self.network.nodes.iter().enumerate() {
            // Check if the node's layer is one we care about
            if layers_to_check.contains(&node.layer_index) {
                // Calculate squared distance (cheaper than sqrt)
                let dist_sq = (node.x - x).powi(2) + (node.y - y).powi(2);

                // Check if within radius AND closer than the current best match
                if dist_sq < selection_radius_sq && dist_sq < closest_node_dist_sq {
                     closest_node_dist_sq = dist_sq;
                     self.hovered_node_idx = Some(idx);
                }
            }
        }
        // We find the *closest* node within the radius in the checked layers.
    }

    fn handle_mouse_click(&mut self, col: u16, row: u16, term_width: u16, term_height: u16) {

        let (x, y) = self.terminal_to_canvas_coords(col, row, term_width, term_height);
        self.handle_mouse_move(col, row, term_width, term_height);
        // Use the hovered node (if any) for selection
        if let Some(idx) = self.hovered_node_idx {
            // Toggle selection - if already selected, deselect it
            if self.selected_node == Some(idx) {
                self.selected_node = None;
            } else {
                self.selected_node = Some(idx);
            }
        } else {
            // Clicking on empty space deselects
            self.selected_node = None;
        }
    }

    fn update_network_layout(&mut self, architecture: &ModelArchitecture) {
        // Convert architecture into layer sizes
        let layer_sizes: Vec<usize> = architecture.layers.iter()
            .map(|layer| match (&layer.input_size, &layer.output_size) {
                (Some(input), _) => input[0],
                (_, Some(output)) => output[0],
                _ => 0 // Skip layers we can't size
            })
            .filter(|&size| size > 0)
            .collect();

        if !layer_sizes.is_empty() {
            self.network = NetworkLayout::new(&layer_sizes);
        }
    }

    fn handle_zmq_update(&mut self, update: Update) {
        log_to_file(&format!("Received ZMQ Update: {:?}", update));
        self.log_recieved_update(&update);

        // Initialize start time on first update if not set
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        match update.type_.as_str() {
            "activation" => {
                if let serde_json::Value::Object(data) = update.data {
                    if let (Some(layer), Some(node), Some(value)) = (
                        data.get("layer").and_then(|v| v.as_u64()),
                        data.get("node").and_then(|v| v.as_u64()),
                        data.get("value").and_then(|v| v.as_f64())
                    ) {
                        if !self.network.nodes.is_empty() {
                            self.network.update_activation(
                                layer as usize,
                                node as usize,
                                value
                            );
                        }
                    }
                }
            },

            "connection" => {
                if let serde_json::Value::Object(data) = update.data {
                    if let Some(from) = data.get("from").and_then(|v| v.as_object()) {
                        if let Some(to) = data.get("to").and_then(|v| v.as_object()) {
                            if let Some(active) = data.get("active").and_then(|v| v.as_bool()) {
                                if !self.network.nodes.is_empty() {
                                    let from_pos = (
                                        from.get("layer").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                                        from.get("node").and_then(|v| v.as_u64()).unwrap_or(0) as usize
                                    );
                                    let to_pos = (
                                        to.get("layer").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                                        to.get("node").and_then(|v| v.as_u64()).unwrap_or(0) as usize
                                    );
                                    self.network.update_connection(from_pos, to_pos, 1.0, active);
                                }
                            }
                        }
                    }
                }
            },

            "layer_state" => {
                if let serde_json::Value::Object(data) = update.data {
                    if let Some(layer_idx) = data.get("layer").and_then(|v| v.as_u64()) {
                        if let Some(activations) = data.get("activations").and_then(|v| v.as_array()) {
                            // Rate limit visualization updates
                            if self.last_viz_update.elapsed() > Duration::from_millis(100) && !self.network.nodes.is_empty() {
                                for (node_idx, val) in activations.iter().enumerate() {
                                    if let Some(value) = val.as_f64() {
                                        self.network.update_activation(
                                            layer_idx as usize,
                                            node_idx,
                                            value
                                        );
                                    }
                                }
                                self.last_viz_update = Instant::now();
                            }
                        }
                    }
                }
            },

            "batch" => {
                if let serde_json::Value::Object(data) = update.data {
                    if let Some(metrics) = data.get("metrics").and_then(|v| v.as_object()) {
                        // Update batch number if available
                        if let Some(batch) = data.get("batch").and_then(|v| v.as_u64()) {
                            self.current_batch = Some(batch as usize);
                        }

                        // Process each metric
                        for (name, value) in metrics.iter() {
                            self.current_metrics.insert(name.clone(), value.clone());

                            // Format and add to output lines for display
                            let display_line = format!("Batch {}: {}: {}",
                                self.current_batch.unwrap_or(0),
                                name,
                                format_value(value)
                            );
                            self.output_lines.push(display_line);
                        }

                        // Keep output lines at a reasonable size
                        if self.output_lines.len() > 1000 {
                            self.output_lines.drain(0..500);
                            if self.training_scroll > 0 {
                                self.training_scroll = self.training_scroll.saturating_sub(500);
                            }
                        }
                        if let Some(metrics) = data.get("metrics").and_then(|v| v.as_object()) {
                           let mut metrics_map = HashMap::new();
                           for (name, value) in metrics {
                               if let Some(val) = value.as_f64() {
                                   metrics_map.insert(name.clone(), val);
                               }
                           }
                           if !metrics_map.is_empty() {
                               self.metrics_history.push(TrainingMetrics {
                                   epoch: self.current_epoch.unwrap_or(0),
                                   metrics: metrics_map,
                               });

                               if self.metrics_history.len() > 1000 {
                                   self.metrics_history.drain(0..500);
                               }
                           }
                       }

                    }
                }
            },

            "epoch" => {
                if let serde_json::Value::Object(data) = update.data {
                    // Update epoch number if available
                    if let Some(epoch) = data.get("epoch").and_then(|v| v.as_u64()) {
                        self.current_epoch = Some(epoch as usize);
                    }

                    if let Some(metrics) = data.get("metrics").and_then(|v| v.as_object()) {
                        let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
                        let epoch_header = format!(
                            "\nEpoch {}/{} [{:02}:{:02}:{:02}]",
                            self.current_epoch.unwrap_or(0),
                            self.total_epochs.unwrap_or(0),
                            elapsed.as_secs() / 3600,
                            (elapsed.as_secs() % 3600) / 60,
                            elapsed.as_secs() % 60
                        );
                        self.output_lines.push(epoch_header.clone());
                        log_to_file(&format!("Added epoch header: {}", epoch_header.clone()));

                        for (name, value) in metrics.iter() {
                            self.current_metrics.insert(name.clone(), value.clone());
                            let display_line = format!("{}: {}", name, format_value(value));
                            self.output_lines.push(display_line.clone());
                            log_to_file(&format!("Added epoch metric: {}", display_line));
                        }

                        // Add a blank line after epoch metrics for better readability
                        self.output_lines.push(String::new());
                    }
                    if let Some(metrics) = data.get("metrics").and_then(|v| v.as_object()) {
                        let mut metrics_map = HashMap::new();

                        for (name, value) in metrics {
                            if let Some(val) = value.as_f64() {
                                metrics_map.insert(name.clone(), val);
                            }
                        }

                        if !metrics_map.is_empty() {
                            self.metrics_history.push(TrainingMetrics{
                                epoch: self.current_epoch.unwrap_or(0),
                                metrics: metrics_map,
                            });
                            if self.metrics_history.len() > 1000 {
                                self.metrics_history.drain(0..500);
                            }
                        }
                    }
                }
            },

            "status" => {
                if let serde_json::Value::Object(data) = update.data {
                    if let Some(state) = data.get("state").and_then(|v| v.as_str()) {
                        match state {
                            "paused" => {
                                self.is_paused = true;
                                self.update_script_state(ScriptState::Paused);
                                self.output_lines.push("Training paused".to_string());
                            },
                            "resumed" => {
                                self.is_paused = false;
                                self.update_script_state(ScriptState::Running);
                                self.output_lines.push("Training resumed".to_string());
                            },
                            "stopped" => {
                                self.update_script_state(ScriptState::Stopped);
                                self.output_lines.push("Training stopped".to_string());
                            },
                            _ => {}
                        }
                    }
                }
            },
            "prediction" => {
                if let serde_json::Value::Object(data) = update.data {
                    if let Some(values_array) = data.get("values").and_then(|v| v.as_array()) {
                        let values: Vec<f64> = values_array.iter()
                            .filter_map(|v| v.as_f64())
                            .collect();

                        // Extract optional labels
                        let labels = data.get("labels")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect::<Vec<String>>()
                            });

                        // Extract description if provided
                        let description = data.get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Model prediction")
                            .to_string();

                        self.set_model_prediction(values, labels, description);
                    }
                }
            },
            _ => {
                log_to_file(&format!("Unknown update type: {}", update.type_));
            }
        }
    }

    fn scroll_training_log(&mut self, delta: i32) {
        let new_scroll = (self.training_scroll as i32 + delta).max(0) as usize;
        let max_scroll = self.output_lines.len().saturating_sub(1);
        self.training_scroll = new_scroll.min(max_scroll);
    }
    fn handle_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => true,
            KeyCode::Char('p') | KeyCode::Enter => {
                log_to_file("Pause/Resume key pressed");
                if let Some(ref mut server) = self.zmq_server {
                    let command = if self.is_paused { "resume" } else { "pause" };
                    match server.send_command(command) {
                        Ok(_) => {
                            self.is_paused = !self.is_paused;
                            log_to_file(&format!("Successfully sent {} command", command));
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to send {} command: {}", command, e);
                            log_to_file(&error_msg);
                            self.log_error(&error_msg);
                        }
                    }
                } else {
                    log_to_file("No ZMQ server available");
                    self.log_error("No ZMQ server available");
                }
                false
            },
            KeyCode::Char('s') => {
                log_to_file("Stop key pressed");
                if let Some(ref mut server) = self.zmq_server {
                    match server.send_command("stop") {
                        Ok(_) => {
                            log_to_file("Stop command sent successfully");
                            self.update_script_state(ScriptState::Stopped)

                        }
                        Err(e) => {
                            let error_msg = format!("Failed to send stop command: {}", e);
                            log_to_file(&error_msg);
                            self.log_error(&error_msg);
                        }
                    }
                } else {
                    log_to_file("No ZMQ server available");
                    self.log_error("No ZMQ server available");
                }
                false
            },
            KeyCode::Char('c') => {
                self.error_log.clear();
                if matches!(self.script_state, ScriptState::Error(_)) {
                    self.script_state = ScriptState::Running;
                }
                false
            }
            KeyCode::Char('h') => {
                self.show_help();
                false
            }
            KeyCode::Char('e') => {
                self.show_error_logs = !self.show_error_logs;
                false
            }

            KeyCode::Up => {
                if self.show_error_logs {
                    self.scroll_error_log(-1);
                } else { self.scroll_training_log(-1); }
                false
            }

            KeyCode::Down => {
                if self.show_error_logs {
                    self.scroll_error_log(1);
                } else { self.scroll_training_log(1); }
                false
            }
            KeyCode::Char('o') => {
                self.toggle_output_view();
                false
            },


            _ => false,
        }
    }

    /*
     * I think these group of functions are very extendable but will need a code refactor later
     * down the line and optimization of the rust code base in general
     *
     * Since these functions point at the batch, epochs, and individual metrics these can probably
     * just be moved around other parts of the ui as needed since there is some redundency that
     * needs to be cleaned up way later
     */
    fn update_metric(&mut self, name: &str, value: serde_json::Value) {
        self.current_metrics.insert(name.to_string(), value.clone());
    }

    fn scroll_error_log(&mut self, delta: i32) {
        let new_scroll = (self.error_scroll as i32 + delta).max(0) as usize;
        self.error_scroll = new_scroll;
    }

    fn show_help(&mut self) {
        self.output_lines.retain(|line| !line.contains("=== Keyboard Controls ==="));
        let help_messages = vec![
            "\n=== Keyboard Controls ===",
            "q/ESC : Quit",
            "p/SPACE: Pause/Resume training",
            "s     : Stop training",
            "e     : Toggle error log",
            "↑/↓   : Scroll error log",
            "c     : Clear error log",
            "h     : Show help",
            "TAB/n : Cycle through nodes",
            "Click : Switch node panel ",
            "======================",
        ];

        for msg in help_messages {
            self.output_lines.push(msg.to_string());
        }
    }

    fn log_error(&mut self, error: &str) {
        if !self.error_log.contains(&error.to_string()) {
            self.error_log.push(error.to_string());
        }
    }

    fn update_script_state(&mut self, state:ScriptState) {
        self.script_state = state.clone();
        if let ScriptState::Error(error) = &state {
            self.log_error(&error.to_string());
        }
        if matches!(state, ScriptState::Completed | ScriptState::Stopped) &&
               !matches!(self.script_state, ScriptState::Completed | ScriptState::Stopped) {
                if let Some(start_time) = self.start_time {
                    self.final_elapsed = Some(start_time.elapsed());
                }
            }
        self.script_state = state;
    }

    fn update_architecture(&mut self, architecture: ModelArchitecture) {
        // Convert architecture into layer sizes
        let layer_sizes: Vec<usize> = architecture.layers.iter()
            .map(|layer| match (&layer.input_size, &layer.output_size) {
                (Some(input), _) => input[0],
                (_, Some(output)) => output[0],
                _ => 0
            })
            .filter(|&size| size > 0)
            .collect();

        // Only update network if we have valid layer sizes
        if !layer_sizes.is_empty() {
            self.network = NetworkLayout::new(&layer_sizes);
        }

        self.model_architecture = architecture;
    }

   fn update_system_metrics(&mut self) {
        self.sys.refresh_all();

        // Calculate CPU usage across all cores
        let cpu_usage = self.sys.global_cpu_info().cpu_usage();

        // Get memory information
        let memory_used = self.sys.used_memory();
        let memory_total = self.sys.total_memory();

        // Try to get GPU information if available
        let gpu_info = get_gpu_info();

        self.system_metrics = Some(SystemMetrics {
            cpu_usage,
            memory_used,
            memory_total,
            gpu_info,
            timestamp: Instant::now(),
        });
    }
}

fn format_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.abs() < 0.0001 || f.abs() >= 10000.0 {
                    format!("{:.2e}", f)
                } else {
                    format!("{:.4}", f)
                }
            } else {
                n.to_string()
            }
        },
        _ => value.to_string(),
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn get_gpu_info() -> Option<GpuInfo> {
     if let Some(nvidia_info) = get_nvidia() {
         return Some(nvidia_info);
     }

     //if let Some(metal_info) = get_metal() {
     //    return Some(metal_info);
     //}

    // TODO add AMD metrics, need an AMD card to test with 
    None
}

fn get_nvidia() -> Option<GpuInfo> {
    let output = StdCommand::new("nvidia-smi")
            .args(&["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;

    if output.status.success() {
        let output_str = String::from_utf8_lossy(&output.stdout);
        let values: Vec<&str> = output_str.trim().split(',').collect();
        if values.len() == 3 {
            return Some(GpuInfo {
                utilization: values[0].trim().parse().unwrap_or(0.0),
                memory_used: values[1].trim().parse().unwrap_or(0),
                memory_total: values[2].trim().parse().unwrap_or(0),

            });
        }
    }
    None
}


/* 
 * Since there is no real API for us to track just going to have to do the best we can if someone
 * is using metal/mps for the backend
 * Honestly was just messing with MPS so I can get used to it before doing any compiler graphs for
 * ONNX + LLVM I really don't care that much about metal or mps until the API gets updated I am not
 * doing the swift dynamic lib unless there is a nightly build because I don't feel like having
 * anyone build the linker library and don't feel like doing anymore FFI currently 
 * switft -> C++ -> rust I cba rn
 * Leave this here for now for skeleton code on any updates to metal api  
 

fn get_metal() -> Option<GpuInfo> {
    #[cfg(target_os = "macos")]
    {   
        // Fallback to Activity Monitor-style metrics
        let ps_output = StdCommand::new("ps")
            .args(&["-A", "-o", "%cpu,command"])
            .output()
            .ok()?;
            
        let ps_str = String::from_utf8_lossy(&ps_output.stdout);
        
        let mut utilization = 0.0;
        let mut seen_processes = 0;
        
        // Look specifically for Metal GPU processes
        for line in ps_str.lines() {
            if contains_metal_process(line) {
                if let Some(cpu_str) = line.split_whitespace().next() {
                    if let Ok(cpu) = cpu_str.parse::<f32>() {
                        // Weight different process types differently
                        let weight = determine_process_weight(line);
                        utilization += cpu * weight;
                        seen_processes += 1;
                    }
                }
            }
        }
        
        // If we didn't find any explicit Metal processes but PyTorch is using MPS
        if seen_processes == 0 && is_mps_backend_active() {
            // Look for Python processes and assume MPS activity
            for line in ps_str.lines() {
                if line.contains("python") {
                    if let Some(cpu_str) = line.split_whitespace().next() {
                        if let Ok(cpu) = cpu_str.parse::<f32>() {
                            utilization += cpu * 0.4; 
                            seen_processes += 1;
                        }
                    }
                }
            }
        }
        
        // Cap and normalize utilization
        utilization = if seen_processes > 0 { 
            (utilization / seen_processes as f32).min(100.0) 
        } else { 
            0.0 
        };

        // Get memory information
        let mut memory_total: u64 = 0;
        let output = StdCommand::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output()
            .ok()?;
            
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            
            // Try to extract VRAM information
            if let Some(vram_line) = output_str.lines().find(|line| line.contains("VRAM")) {
                if let Some(vram_str) = vram_line.split(':').nth(1) {
                    let vram_str = vram_str.trim();
                    if let Some(gb_pos) = vram_str.find("GB") {
                        if let Ok(gb) = vram_str[..gb_pos].trim().parse::<f32>() {
                            memory_total = (gb * 1024.0) as u64; // Convert GB to MB
                        }
                    } else if let Some(mb_pos) = vram_str.find("MB") {
                        if let Ok(mb) = vram_str[..mb_pos].trim().parse::<u64>() {
                            memory_total = mb;
                        }
                    }
                }
            }
            
            // For Apple Silicon, check for unified memory
            if memory_total == 0 && output_str.contains("Apple M") {
                // Default to 4GB for Apple Silicon GPUs if no specific info
                memory_total = 4 * 1024; 
            }
        }
        
        // Cap and normalize utilization
        utilization = if seen_processes > 0 { 
            (utilization / seen_processes as f32).min(100.0) 
        } else { 
            0.0 
        };
        
        // Estimate memory usage from utilization
        let memory_used = ((memory_total as f32) * (utilization / 100.0)) as u64;


        
        return Some(GpuInfo {
            utilization,
            memory_used,
            memory_total,
        });
    }
    #[cfg(not(target_os = "macos"))]
    None
}
*/
// Helper functions 
fn contains_metal_process(line: &str) -> bool {
    line.contains("Metal") || 
    line.contains("MTL") || 
    line.contains("GPU") || 
    line.contains("AccelerationEngine") ||
    line.contains("GPUWorker")
}

fn determine_process_weight(line: &str) -> f32 {
    if line.contains("python") && line.contains("mps") {
        0.8 // High likelihood of PyTorch MPS usage
    } else if line.contains("MetalWorker") {
        0.9 // Direct Metal worker processes
    } else if line.contains("GPU") {
        0.85 // Generic GPU processes
    } else {
        0.5 // Other potential Metal-related processes
    }
}

fn is_mps_backend_active() -> bool {
    // Check for MPS environment variables or known indicators
    std::env::var("PYTORCH_ENABLE_MPS_FALLBACK").is_ok() ||
    std::env::var("PYTORCH_MPS_HIGH_WATERMARK_RATIO").is_ok()
}

fn get_metal_memory_info(gpu_model: Option<String>) -> (u64, u64) {
    // IOKit can provide better memory information but requires building a swift dynamic lib 
    // Don't care to do that right now for metal, rather just wait for API updates 
    // For now, approximate based on device model_prediction

    match gpu_model.as_deref() {
        Some(model) if model.contains("M1 Pro") => (2 * 1024, 16 * 1024),
        Some(model) if model.contains("M1 Max") => (4 * 1024, 32 * 1024),
        Some(model) if model.contains("M1 Ultra") => (8 * 1024, 64 * 1024),
        Some(model) if model.contains("M2") => (3 * 1024, 24 * 1024),
        Some(model) if model.contains("M3") => (4 * 1024, 32 * 1024),
        _ => (2 * 1024, 8 * 1024), // Default fallback
    }
}





fn render_system_metrics(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title("System Metrics")
        .borders(Borders::ALL);
    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if let Some(metrics) = &app.system_metrics {
        let mem_percentage = (metrics.memory_used as f64 / metrics.memory_total as f64 * 100.0) as u64;

        let mut text = format!(
            "\nCPU Usage: {:.1}%\n\
             Memory: {} / {} ({:.1}%)",
            metrics.cpu_usage,
            format_bytes(metrics.memory_used * 1024), // Convert KB to bytes
            format_bytes(metrics.memory_total * 1024),
            mem_percentage,
        );

        // Add GPU metrics if available
        if let Some(gpu) = &metrics.gpu_info {
            text.push_str(&format!(
                "\n\nGPU:\n\
                 Utilization: {:.1}%\n\
                 Memory: {} / {}",
                gpu.utilization,
                format_bytes(gpu.memory_used * 1024 * 1024), // Convert MB to bytes
                format_bytes(gpu.memory_total * 1024 * 1024),
            ));
        }

        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(Color::White));
        f.render_widget(paragraph, inner_area);
    }
}

fn render_training_progress(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(if app.show_error_logs {
            "Error Log"
        } else {
            "Training Progress"
        })
        .title_style(Style::default().fg(match &app.script_state {
            ScriptState::Error(_) => Color::Red,
            _ => Color::White,
        }))
        .borders(Borders::ALL);

    let inner_area = block.inner(area);
    f.render_widget(block, area);

    let text = if app.show_error_logs {
        app.error_log.iter()
            .skip(app.error_scroll)
            .map(|err| format!("❌ {}", err))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        app.output_lines.iter()
            .skip(app.training_scroll)
            .cloned()
            .collect::<Vec<String>>()
            .join("\n")
    };

    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(match &app.script_state {
            ScriptState::Error(_) => Color::Red,
            _ => Color::White,
        }));

    let margin = Margin {
        vertical: 1,
        horizontal: 1,
    };
    f.render_widget(paragraph, inner_area.inner(margin));
}


fn render_node_info(f: &mut Frame, app: &App, area: Rect) {
    let node_block = Block::default()
        .title("Node Information")
        .borders(Borders::ALL);

    let inner_area = node_block.inner(area);
    f.render_widget(node_block, area);

    if app.network.nodes.is_empty() || app.selected_node.is_none() {
        let text = "No node selected\nUse tab key to select nodes";
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));
        f.render_widget(paragraph, inner_area);
        return;
    }

    if let Some(node_idx) = app.selected_node {
        if node_idx < app.network.nodes.len() {
            let node = &app.network.nodes[node_idx];

            // Count input and output connections
            let input_conn_count = app.network.connections.iter()
                .filter(|c| c.to_node_id == node.id)
                .count();

            let output_conn_count = app.network.connections.iter()
                .filter(|c| c.from_node_id == node.id)
                .count();

            let node_type = match node.node_type {
                NodeType::Input => "Input",
                NodeType::Hidden => "Hidden",
                NodeType::Output => "Output",
            };

            let node_info = vec![
                Line::from(vec![
                    Span::raw("Layer: "),
                    Span::styled(format!("{}", node.layer_index), Style::default().fg(Color::Cyan))
                ]),
                Line::from(vec![
                    Span::raw("Index: "),
                    Span::styled(format!("{}", node.original_index), Style::default().fg(Color::White))
                ]),
                Line::from(vec![
                    Span::raw("Type: "),
                    Span::styled(node_type, Style::default().fg(match node.node_type {
                        NodeType::Input => Color::Blue,
                        NodeType::Hidden => Color::White,
                        NodeType::Output => Color::Green,
                    }))
                ]),
                Line::from(vec![
                    Span::raw("Activation: "),
                    Span::styled(
                        format!("{:.4}", node.activation.unwrap_or(0.0)),
                        Style::default().fg(
                            if node.activation.unwrap_or(0.0) > 0.5 {
                                Color::Green
                            } else {
                                Color::White
                            }
                        )
                    )
                ]),
                Line::from(vec![
                    Span::raw("Connections: "),
                    Span::styled(format!("{} in, {} out", input_conn_count, output_conn_count),
                        Style::default().fg(Color::White))
                ]),
            ];

            let paragraph = Paragraph::new(node_info)
                .alignment(Alignment::Left);

            f.render_widget(paragraph, inner_area);
        }
    } else {
        // No node selected
        let text = "No node selected\nUse Tab key to select nodes";
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));

        f.render_widget(paragraph, inner_area);
    }
}

fn render_metrics(f: &mut Frame, app: &mut App, area: Rect) {
    let metrics_block = Block::default()
        .title("Training Status")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(match &app.script_state {
            ScriptState::Error(_) => Color::Red,
            ScriptState::Completed => Color::Green,
            ScriptState::Stopped => Color::Yellow,
            _ => if app.is_paused {Color::Yellow} else {Color::White},
        }));

    let inner_area = metrics_block.inner(area);
    f.render_widget(metrics_block, area);

    // Build status section
    let mut text = Vec::new();

    // Training status with color
    let status_text = match &app.script_state {
        ScriptState::Starting => Span::styled("Starting", Style::default().fg(Color::Blue)),
        ScriptState::Running => {
            if app.is_paused {
                Span::styled("Paused", Style::default().fg(Color::Yellow))
            } else {
                Span::styled("Running", Style::default().fg(Color::Green))
            }
        },
        ScriptState::Error(_) => Span::styled("Error", Style::default().fg(Color::Red)),
        ScriptState::Completed => Span::styled("Complete", Style::default().fg(Color::Green)),
        ScriptState::Stopped => Span::styled("Stopped", Style::default().fg(Color::LightRed)),
        ScriptState::Paused => Span::styled("Paused", Style::default().fg(Color::Yellow)),
    };

    text.push(Line::from(vec![
        Span::raw("Status: "),
        status_text
    ]));

    // Framework info
    let framework_text = match &app.model_architecture.framework {
        Some(MLFramework::PyTorch) => "PyTorch",
        Some(MLFramework::TensorFlow) => "TensorFlow",
        Some(MLFramework::JAX) => "JAX",
        Some(MLFramework::Keras) => "Keras",
        Some(MLFramework::Unknown) => "Unknown",
        None => "Not Detected",
    };
    text.push(Line::from(vec![
        Span::raw("Framework: "),
        Span::styled(framework_text, Style::default().fg(Color::Cyan))
    ]));

    // Add model summary
    text.push(Line::from(""));
    text.push(Line::from("Model Summary:"));

    let total_params = app.model_architecture.total_parameters;
    let param_text = if total_params > 1_000_000 {
        format!("{:.2}M", total_params as f64 / 1_000_000.0)
    } else if total_params > 1_000 {
        format!("{:.2}K", total_params as f64 / 1_000.0)
    } else {
        format!("{}", total_params)
    };

    let layer_count = app.model_architecture.layers.len();

    text.push(Line::from(vec![
        Span::raw("Layers: "),
        Span::styled(format!("{}", layer_count), Style::default().fg(Color::White))
    ]));

    text.push(Line::from(vec![
        Span::raw("Parameters: "),
        Span::styled(param_text, Style::default().fg(Color::White))
    ]));

    // Add progress information if available
    if let (Some(epoch), Some(total_epochs)) = (app.current_epoch, app.total_epochs) {
        let progress = (epoch as f64 / total_epochs as f64 * 100.0).round() as usize;
        text.push(Line::from(""));
        text.push(Line::from(vec![
            Span::raw("Progress: "),
            Span::styled(
                format!("Epoch {}/{} ({}%)", epoch, total_epochs, progress),
                Style::default().fg(Color::Green)
            )
        ]));
    }

    if let Some(start_time) = app.start_time { //TODO fix timer for paused state
        let elapsed = match app.script_state {
            ScriptState::Completed | ScriptState::Stopped => {
                static mut FINAL_TIME: Option<Duration> = None;

                unsafe {
                    if FINAL_TIME.is_none() {
                        FINAL_TIME = Some(start_time.elapsed());
                    }
                    FINAL_TIME.unwrap_or(start_time.elapsed())
                }
            },
            ScriptState::Paused => {
                app.paused_elapsed.unwrap_or_else(|| start_time.elapsed())
            }
            _ => {
                let current_elapsed = start_time.elapsed();
                app.paused_elapsed = Some(current_elapsed);
                current_elapsed
            }
        };

        // Format with or without "(final)" tag
        let time_text = match app.script_state {
            ScriptState::Completed | ScriptState::Stopped => {
                format!(
                    "{:02}:{:02}:{:02} (final)",
                    elapsed.as_secs() / 3600,
                    (elapsed.as_secs() % 3600) / 60,
                    elapsed.as_secs() % 60
                )
            },
            _ => {
                format!(
                    "{:02}:{:02}:{:02}",
                    elapsed.as_secs() / 3600,
                    (elapsed.as_secs() % 3600) / 60,
                    elapsed.as_secs() % 60
                )
            }
        };

        text.push(Line::from(vec![
            Span::raw("Training Time: "),
            Span::styled(time_text, Style::default().fg(Color::White))
        ]));
    }



    // Add current metrics
    text.push(Line::from(""));
    text.push(Line::from(vec![
        Span::styled("Current Metrics:", Style::default().fg(Color::White))
    ]));

    for (name, value) in &app.current_metrics {
        text.push(Line::from(vec![
            Span::raw(format!("{}: ", name)),
            Span::styled(format_value(value),
                Style::default().fg(if name == "loss" { Color::Red } else { Color::Green }))
        ]));
    }

    let paragraph = Paragraph::new(text)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    f.render_widget(paragraph, inner_area);
    }



fn render_metrics_chart(f: &mut Frame, app: &App, area: Rect) {
    let chart_block = Block::default()
        .title("Training Metrics")
        .borders(Borders::ALL);

    let inner_area = chart_block.inner(area);
    f.render_widget(chart_block, area);

    // Skip rendering if no data yet
    if app.metrics_history.is_empty() || app.current_metrics.is_empty() {
        let text = "Waiting for training data";
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));
        f.render_widget(paragraph, inner_area);
        return;
    }

    // Get a list of all available metrics from current metrics
    let available_metrics: Vec<String> = app.current_metrics.keys()
        .cloned()
        .collect();

    // Limit to 5 metrics maximum to avoid cluttering the chart
    let metrics_to_show = available_metrics.iter()
        .take(5)
        .cloned()
        .collect::<Vec<String>>();

    if metrics_to_show.is_empty() {
        let text = "No metrics available to plot";
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));
        f.render_widget(paragraph, inner_area);
        return;
    }

    // Define a set of distinct colors for the metrics
    let colors = [
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
    ];

    // Store all data outside the loop so it lives long enough
    let mut all_data: Vec<(String, Vec<(f64, f64)>, Color)> = Vec::new();
    let mut max_value: f64 = 0.0;
    let mut min_value: f64 = f64::MAX;
    let max_points = app.metrics_history.len() as f64;

    // Collect data for all metrics
    for (idx, metric_name) in metrics_to_show.iter().enumerate() {
        // Get color (cycle through colors if we have more metrics than colors)
        let color = colors[idx % colors.len()];

        // Extract data for this metric from history
        let data: Vec<(f64, f64)> = app.metrics_history.iter()
            .enumerate()
            .filter_map(|(i, m)| {
                m.metrics.get(metric_name).map(|&val| (i as f64, val))
            })
            .collect();

        // Skip empty datasets
        if data.is_empty() {
            continue;
        }

        // Update min/max values for scaling
        for &(_, value) in &data {
            max_value = max_value.max(value);
            min_value = min_value.min(value);
        }

        // Store the data for later use
        all_data.push((metric_name.clone(), data, color));
    }


    if all_data.is_empty() {
        let text = "No plottable metrics history available";
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));
        f.render_widget(paragraph, inner_area);
        return;
    }

    // Ensure min_value is not greater than max_value
    if min_value > max_value {
        min_value = 0.0;
    }

    // Add a small margin to the bounds
    let bound_margin = (max_value - min_value) * 0.1;
    let y_min = (min_value - bound_margin).max(0.0);  // Don't go below zero unless values are negative
    let y_max = max_value + bound_margin;


    let datasets: Vec<Dataset> = all_data.iter()
        .map(|(name, data, color)| {
            Dataset::default()
                .name(name.as_str())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(*color))
                .data(data)
        })
        .collect();

    // Render chart with proper scaling
    let chart = Chart::new(datasets)
        .x_axis(Axis::default()
            .title("Steps")
            .style(Style::default().fg(Color::Gray))
            .bounds([0.0, max_points])
            .labels(vec!["0".to_string(), format!("{}", max_points as usize)]))
        .y_axis(Axis::default()
            .title("Value")
            .style(Style::default().fg(Color::Gray))
            .bounds([y_min, y_max])
            .labels(vec![
                format!("{:.2}", y_min),
                format!("{:.2}", (y_min + y_max) / 2.0),
                format!("{:.2}", y_max)
            ]));

    f.render_widget(chart, inner_area);
}

fn render_layer_info(f: &mut Frame, app: &App, area: Rect) {
    let layers_block = Block::default()
        .title("Network Layers")
        .borders(Borders::ALL);

    let inner_area = layers_block.inner(area);
    f.render_widget(layers_block, area);

    let mut layer_texts = Vec::new();
    let mut total_params = 0;

    for (idx, layer) in app.model_architecture.layers.iter().enumerate() {
        let layer_stats = format!(
            "{}: {} ({})",
            idx,
            layer.layer_type,
            format_params(layer.parameters)
        );
        layer_texts.push(layer_stats);
        total_params += layer.parameters;
    }

    layer_texts.push(format!("Total params: {}", format_params(total_params)));

    let paragraph = Paragraph::new(layer_texts.join("\n"))
        .alignment(Alignment::Left);

    f.render_widget(paragraph, inner_area);
}

fn format_params(params: usize) -> String {
    if params < 1_000 {
        format!("{}", params)
    } else if params < 1_000_000 {
        format!("{:.2}K", params as f64 / 1_000.0)
    } else {
        format!("{:.2}M", params as f64 / 1_000_000.0)
    }
}

fn render_model_output(f: &mut Frame, app: &App, area: Rect) {
    let output_block = Block::default()
        .title("Model Prediction")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner_area = output_block.inner(area);
    f.render_widget(output_block, area);

    if let Some(prediction) = &app.model_prediction {
        let mut lines = Vec::new();

        // Add header information with better styling
        lines.push(Line::from(vec![
            Span::styled(
                prediction.description.clone(),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            )
        ]));

        lines.push(Line::from(vec![
            Span::styled(format!("Epoch {}", prediction.epoch),
                Style::default().fg(Color::Cyan))
        ]));

        let time_display = match app.script_state { //TODO Seperate these based on match case
            ScriptState::Paused | ScriptState::Stopped | ScriptState::Error(_) => {
                "Stopped".to_string()
            }
            ScriptState::Completed => { // TODO Format this Time Completed: {time}
                "Completed".to_string()
            }
            _ => {
                // If running, calculate the time difference
                let time = Instant::now()
                    .duration_since(prediction.timestamp)
                    .as_secs_f32();
                format!("{:.1}s ago", time)
            }

        };

        lines.push(Line::from(vec![
            Span::raw("Time: "),
            Span::styled(
                time_display,
                Style::default().fg(Color::Gray),
            ),
        ]));


        lines.push(Line::from(""));

        // Get the highest probability for coloring
        let max_value = prediction.values.iter()
            .fold(0.0f64, |max, &val| max.max(val));

        // Format values
        if let Some(labels) = &prediction.labels {
            // Create a prediction table
            lines.push(Line::from(vec![
                Span::styled("Class Predictions:",
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
            ]));

            lines.push(Line::from(vec![
                Span::styled("Class".to_string(),
                    Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" | "),
                Span::styled("Probability".to_string(),
                    Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" | "),
                Span::styled("Bar".to_string(),
                    Style::default().add_modifier(Modifier::BOLD)),
            ]));

            lines.push(Line::from("-".repeat(inner_area.width as usize - 4)));

            // Create sorted indices for displaying highest probabilities first
            let mut indices: Vec<usize> = (0..prediction.values.len()).collect();
            indices.sort_by(|&i, &j| prediction.values[j].partial_cmp(&prediction.values[i]).unwrap());

            for &i in indices.iter().take(20) {
                let value = prediction.values[i];
                let label = &labels[i];

                // Determine color based on probability
                let color = if value > 0.5 {
                    Color::Green
                } else if value > 0.2 {
                    Color::Yellow
                } else {
                    Color::Gray
                };

                // Create a bar visualization
                let bar_width = ((inner_area.width as f64 - 30.0) * value).round() as usize;
                let bar = "█".repeat(bar_width);

                lines.push(Line::from(vec![
                    Span::styled(format!("{:<10}", label), Style::default().fg(color)),
                    Span::raw(" | "),
                    Span::styled(format!("{:.4}", value), Style::default().fg(color)),
                    Span::raw(" | "),
                    Span::styled(bar, Style::default().fg(color)),
                ]));
            }
        } else {
            // Just show values if no labels
            lines.push(Line::from("Output Values:"));

            for (i, value) in prediction.values.iter().enumerate().take(20) {
                let bar_width = ((inner_area.width as f64 - 20.0) * (*value / max_value)).round() as usize;
                let bar = "█".repeat(bar_width);

                lines.push(Line::from(vec![
                    Span::raw(format!("{:<3}: ", i)),
                    Span::styled(format!("{:.4}", value),
                        Style::default().fg(Color::White)),
                    Span::raw(" "),
                    Span::styled(bar, Style::default().fg(Color::Cyan)),
                ]));
            }
        }

        // If there are more values than we're showing
        if prediction.values.len() > 20 {
            lines.push(Line::from(vec![
                Span::raw(format!("... and {} more values", prediction.values.len() - 20))
            ]));
        }

        // Add help text
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Press 'o' to return to main view",
                Style::default().fg(Color::Gray))
        ]));

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, inner_area);
    } else {
        let text = "No model prediction captured yet.\n\nPredictions will appear after your model runs inference.";
        let paragraph = Paragraph::new(text)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));

        f.render_widget(paragraph, inner_area);
    }
}

fn render_layout(f: &mut Frame, app: &mut App) {
    if app.show_model_output {
        let output_area = f.area();
        render_model_output(f, app, output_area);
        return;
    }
    let terminal_size = f.area();
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60),  // Top section
            Constraint::Percentage(40),  // Bottom section
        ])
        .split(f.area());

    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70),  // Left - model visualization
            Constraint::Percentage(30),  // Right - metrics panel
        ])
        .split(main_chunks[0]);

    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),  // Left - metrics plot
            Constraint::Percentage(50),  // Right - log/system metrics
        ])
        .split(main_chunks[1]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),  // Top - training log
            Constraint::Percentage(50),  // Bottom - system metrics
        ])
        .split(bottom_chunks[1]);

    // Render the network diagram
    let network_canvas = app.network.draw();
    f.render_widget(
        network_canvas.block(
            Block::default()
                .title("Model Architecture")
                .borders(Borders::ALL)
        ),
        top_chunks[0]
    );

    // Render layer info and metrics
    render_metrics(f, app, top_chunks[1]);

    // If a node is selected, show details
    if app.selected_node.is_some() {
        render_node_info(f, app, bottom_chunks[0]);
    } else {
        // Show metrics plot when no node selected
        render_metrics_chart(f, app, bottom_chunks[0]);
    }

    // Training log and system metrics
    render_training_progress(f, app, right_chunks[0]);
    render_system_metrics(f, app, right_chunks[1]);

}

fn run_app(python: PythonRunner) -> Result<()> {
    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and initialize architecture
    let mut app = App::new();
    if let Some(arch) = python.get_architecture() {
        app.update_architecture(arch.clone());
    }

    // Setup ZMQ channels
    let (mut zmq_server, update_rx, command_tx) = ZMQServer::new()?;

    // Create a separate channel for updates
    let (update_tx, _update_rx) = mpsc::channel::<Update>();

    // Start the metrics listener with the update channel
    zmq_server.start_listening(update_tx)?;

    app.command_tx = Some(command_tx);
    app.zmq_server = Some(zmq_server);

    let mut last_render = Instant::now();
    //let render_interval = Duration::from_millis(16); //60 fps
    let render_interval = Duration::from_millis(33); // 30 fps
    let mut frame_counter = 0;
    let mut has_error = false;

    let mut last_metrics_update = Instant::now();
    let mut has_error = false;

    loop {
        // Process all pending ZMQ updates
        while let Ok(update) = update_rx.try_recv() {
            log_to_file(&format!("Main loop received update: {:?}", update));
            app.handle_zmq_update(update);
        }

        // Check for Python process state
        match python.receive()? {
            ScriptOutput::Error(error) => {
                app.log_error(&error.to_string());
                app.update_script_state(ScriptState::Error(error));
            }
            ScriptOutput::Terminated => {
                if !matches!(app.script_state, ScriptState::Error(_)) &&
                   !matches!(app.script_state, ScriptState::Stopped) {
                    app.update_script_state(ScriptState::Completed);
                }
            }
            _ => {}
        }

        // Update system metrics every second
        if last_metrics_update.elapsed() >= Duration::from_secs(1) {
            app.update_system_metrics();
            last_metrics_update = Instant::now();
        }

        // Handle input with a short timeout
        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                if app.handle_key(key.code) {
                    break;
                }
            }
        }
        if event::poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Mouse(mouse) => {
                    let terminal_size = terminal.size()?;
                    match mouse.kind {
                        // No handling for MouseEventKind::Moved
                        event::MouseEventKind::Down(event::MouseButton::Left) => {
                            // Handle click selection
                            app.handle_mouse_click(
                                mouse.column,
                                mouse.row,
                                terminal_size.width,
                                terminal_size.height
                            );
                        },
                        event::MouseEventKind::ScrollDown => {
                            // Scroll down in logs or other scrollable elements
                            if app.show_error_logs {
                                app.scroll_error_log(1);
                            } else {
                                app.scroll_training_log(1);
                            }
                        },
                        event::MouseEventKind::ScrollUp => {
                            // Scroll up in logs or other scrollable elements
                            if app.show_error_logs {
                                app.scroll_error_log(-1);
                            } else {
                                app.scroll_training_log(-1);
                            }
                        },
                        event::MouseEventKind::Moved => {
                             app.handle_mouse_move(
                                mouse.column,
                                mouse.row,
                                terminal_size.width,
                                terminal_size.height
                            );                           
                        },

                        _ => {}
                    }
                },
                // Add keyboard navigation for nodes
                Event::Key(key) => {
                    match key.code {
                        KeyCode::Tab => {
                            // Cycle through nodes
                            if app.network.nodes.is_empty() {
                                app.selected_node = None;
                            } else {
                                let next = match app.selected_node {
                                    None => Some(0),
                                    Some(current) => {
                                        if current + 1 < app.network.nodes.len() {
                                            Some(current + 1)
                                        } else {
                                            None // Cycle back to no selection
                                        }
                                    }
                                };
                                app.selected_node = next;
                            }
                        },
                        KeyCode::Char('n') => {
                            // Next node in same layer
                            if let Some(current_idx) = app.selected_node {
                                if current_idx < app.network.nodes.len() {
                                    let current = &app.network.nodes[current_idx];
                                    let layer = current.layer_index;

                                    // Find next node in same layer
                                    let next = app.network.nodes.iter().enumerate()
                                        .filter(|(_, n)| n.layer_index == layer && n.id > current.id)
                                        .map(|(i, _)| i)
                                        .next();

                                    if let Some(next_idx) = next {
                                        app.selected_node = Some(next_idx);
                                    }
                                }
                            }
                        },
                        _ => {
                            if app.handle_key(key.code) {
                                break;
                            }
                        }
                    }
                },
                _ => {}
            }
        }

        // Render frame at 60 FPS
        if last_render.elapsed() >= render_interval {
            terminal.draw(|f| render_layout(f, &mut app))?;
            last_render = Instant::now();
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())


}



fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.debug {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .format_timestamp(None)
            .format_target(false)
            .init();
    } else {
        // Use std::env::temp_dir() for OS-agnostic temp directory
        let mut log_path = std::env::temp_dir();
        log_path.push("aliyah_rust.log");
        let log_file = std::fs::File::create(log_path).unwrap_or_else(|_| {
            // Handle null file in an OS-agnostic way
            if cfg!(windows) {
                std::fs::File::create("NUL").unwrap()
            } else {
                std::fs::File::create("/dev/null").unwrap()
            }
        });
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Error)
            .format_timestamp(None)
            .format_target(false)
            .target(env_logger::Target::Pipe(Box::new(log_file)))
            .init();
    }
    let python = PythonRunner::new(cli.script, cli.script_args)?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let result = run_app(python);
    result
}
