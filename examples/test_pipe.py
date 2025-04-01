import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from aliyah import monitor, trainingmonitor

# 1. Deep Belief Network with RBM layers
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        
        # Register hooks to monitor activations
        self.register_buffer('activation_h', torch.zeros(n_hidden))
        self.register_buffer('activation_v', torch.zeros(n_visible))
        
    def free_energy(self, v):
        """Calculate free energy"""
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W) + self.h_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        return -hidden_term - vbias_term
        
    def sample_h(self, v):
        activation = torch.matmul(v, self.W) + self.h_bias
        h_prob = torch.sigmoid(activation)
        h_prob = torch.clamp(h_prob, 0, 1)
        h_sample = torch.bernoulli(h_prob)
        
        # Store activation for monitoring
        self.activation_h = h_prob.mean(0).detach()
        
        return h_prob, h_sample
    
    def sample_v(self, h):
        activation = torch.matmul(h, self.W.t()) + self.v_bias
        v_prob = torch.sigmoid(activation)
        v_prob = torch.clamp(v_prob, 0, 1)
        v_sample = torch.bernoulli(v_prob)
        
        # Store activation for monitoring
        self.activation_v = v_prob.mean(0).detach()
        
        return v_prob, v_sample

class DBN(nn.Module):
    def __init__(self, layers):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList([
            RBM(layers[i], layers[i+1]) 
            for i in range(len(layers)-1)
        ])
        
        # Store layer dimensions for visualization
        self.layer_dims = layers
    
    def forward(self, x):
        """Forward pass for feature extraction"""
        current = x
        for rbm in self.rbm_layers:
            current, _ = rbm.sample_h(current)
        return current

    def pretrain(self, dataloader, epochs=10, lr=0.001, monitor=None):  
        """Layer-wise pretraining of RBMs with Aliyah monitoring"""
        device = next(self.parameters()).device
        total_epochs = epochs * len(self.rbm_layers)
        current_epoch = 0
        
        # First, create a visualization of the DBN structure
        if monitor:
            # Create visualization nodes for all layers
            for layer_idx, dim in enumerate(self.layer_dims):
                # Limit visualization to max 10 nodes per layer
                vis_size = min(dim, 10)
                activations = [0.5] * vis_size  # Default activation of 0.5
                monitor.log_layer_state(layer_idx, activations)
                
            # Create connections between layers
            for i in range(len(self.layer_dims) - 1):
                # Create sample connections between layers
                for j in range(min(5, self.layer_dims[i])):
                    for k in range(min(5, self.layer_dims[i+1])):
                        # All connections initially active for visualization
                        monitor.log_connection_flow(i, j, i+1, k, True)
        
        for i, rbm in enumerate(self.rbm_layers):
            print(f"Pre-training RBM layer {i+1}/{len(self.rbm_layers)}")
            optimizer = torch.optim.Adam(rbm.parameters(), lr=lr)
            
            for epoch in range(epochs):
                if monitor and not monitor.check_control():
                    print("Training stopped by user")
                    return
                
                total_loss = 0
                batch_count = 0
                
                for batch_idx, batch_data in enumerate(dataloader):
                    if monitor and not monitor.check_control():
                        return
                    
                    batch_data = batch_data.to(device)
                    
                    # Get current input
                    current_input = batch_data
                    if i > 0:
                        with torch.no_grad():
                            for j in range(i):
                                current_input, _ = self.rbm_layers[j].sample_h(current_input)
                    
                    # Positive phase
                    pos_hidden_prob, pos_hidden = rbm.sample_h(current_input)
                    
                    # Negative phase
                    neg_visible_prob, neg_visible = rbm.sample_v(pos_hidden)
                    neg_hidden_prob, neg_hidden = rbm.sample_h(neg_visible_prob)
                    
                    # Calculate gradients using contrastive divergence
                    positive_grad = torch.matmul(current_input.t(), pos_hidden_prob)
                    negative_grad = torch.matmul(neg_visible_prob.t(), neg_hidden_prob)
                    
                    weight_update = positive_grad - negative_grad
                    
                    optimizer.zero_grad()
                    
                    rbm.W.grad = -(weight_update) / batch_data.size(0)
                    rbm.v_bias.grad = -torch.mean(current_input - neg_visible_prob, dim=0)
                    rbm.h_bias.grad = -torch.mean(pos_hidden_prob - neg_hidden_prob, dim=0)
                    
                    optimizer.step()
                    
                    # Calculate loss using free energy difference
                    loss = torch.mean(rbm.free_energy(current_input)) - torch.mean(rbm.free_energy(neg_visible_prob))
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Log batch metrics
                    if monitor and batch_idx % 10 == 0:  # Reduce visualization frequency
                        # Log batch metrics
                        monitor.log_batch(
                            batch_idx, 
                            loss=loss.item(),
                            layer=i
                        )
                        
                        # Get activation values
                        h_activations = rbm.activation_h.cpu().numpy().tolist()
                        v_activations = rbm.activation_v.cpu().numpy().tolist()
                        
                        # Keep lists to a reasonable size for visualization
                        v_activations = v_activations[:min(10, len(v_activations))]
                        h_activations = h_activations[:min(10, len(h_activations))]
                            
                        # Log layer activations for visualization
                        monitor.log_layer_state(i, v_activations)
                        monitor.log_layer_state(i+1, h_activations)
                        
                        # Log some connections with their weights
                        for j in range(min(5, len(v_activations))):
                            for k in range(min(5, len(h_activations))):
                                if j < rbm.W.shape[0] and k < rbm.W.shape[1]:
                                    weight = float(rbm.W[j, k].item())
                                    active = abs(weight) > 0.01
                                    monitor.log_connection_flow(i, j, i+1, k, active)
                
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                
                if monitor:
                    monitor.log_epoch(current_epoch, avg_loss)
                
                current_epoch += 1
                
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

# 2. Particle Swarm Optimization (adapted for monitoring)
class ParticleSwarmOptimizer:
    def __init__(self, n_particles, n_dimensions, bounds, monitor=None):
        self.n_particles = n_particles
        self.bounds = bounds
        self.positions = np.random.randint(bounds[0], bounds[1], (n_particles, n_dimensions))
        self.velocities = np.zeros((n_particles, n_dimensions))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([float('-inf')] * n_particles)
        self.gbest_position = self.positions[0].copy()
        self.gbest_score = float('-inf')
        self.monitor = monitor
        
        # PSO parameters from the paper
        self.w = 0.8  # inertia weight
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
    
    def optimize(self, fitness_func, max_iter=50):
        print("Starting PSO optimization")
        
        # We'll use these layers to visualize PSO as a network
        # Particles as input layer, dimensions as hidden layer, and fitness as output
        if self.monitor:
            # Visualize particles
            for i in range(self.n_particles):
                # Show each particle as a node
                activation = 0.5  # Default activation
                self.monitor.log_layer_activation(0, i, activation)
            
            # Visualize dimensions
            for i in range(len(self.gbest_position)):
                # Show dimensions as middle layer nodes
                self.monitor.log_layer_activation(1, i, 0.5)
            
            # Visualize fitness (output layer, single node)
            self.monitor.log_layer_activation(2, 0, 0.0)
        
        for iteration in range(max_iter):
            if self.monitor and not self.monitor.check_control():
                print("PSO optimization stopped by user")
                return self.gbest_position
                
            if (iteration + 1) % 10 == 0:
                print(f"PSO Iteration {iteration+1}/{max_iter}")
            
            batch_metrics = {}
            
            for i in range(self.n_particles):
                # Evaluate fitness for this particle
                score = fitness_func(self.positions[i])
                
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                    if score > self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = self.positions[i].copy()
                        
                        # Log the best configuration found so far
                        if self.monitor:
                            # Update visualization based on best particle
                            self.monitor.log_layer_activation(2, 0, min(1.0, score / 10.0))  # Normalize fitness
                
                # Visualize this particle's state
                if self.monitor:
                    # Activation of this particle based on its relative fitness
                    rel_fitness = (score - min(self.pbest_scores)) / (max(self.pbest_scores) - min(self.pbest_scores) + 1e-10)
                    self.monitor.log_layer_activation(0, i, min(1.0, rel_fitness))
            
            r1, r2 = np.random.rand(2)
            for i in range(self.n_particles):
                old_velocity = self.velocities[i].copy()
                
                # Update velocity
                self.velocities[i] = (self.w * self.velocities[i] + 
                                    self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                    self.c2 * r2 * (self.gbest_position - self.positions[i]))
                
                # Update position
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i].astype(int),
                    self.bounds[0], self.bounds[1]
                )
                
                # Visualize connections between particles and dimensions
                if self.monitor:
                    for j in range(min(5, len(self.positions[i]))):
                        # Show connections based on velocity
                        active = abs(self.velocities[i][j]) > 0.1
                        self.monitor.log_connection_flow(0, i, 1, j, active)
                        
                        # Show connection from dimension to fitness
                        diff = abs(self.positions[i][j] - self.gbest_position[j])
                        norm_diff = min(1.0, diff / (self.bounds[1] - self.bounds[0]))
                        self.monitor.log_connection_flow(1, j, 2, 0, norm_diff > 0.3)
            
            # Log batch metrics
            if self.monitor:
                batch_metrics = {
                    'best_fitness': self.gbest_score,
                    'avg_fitness': np.mean(self.pbest_scores),
                    'velocity_magnitude': np.mean(np.linalg.norm(self.velocities, axis=1))
                }
                self.monitor.log_batch(iteration, **batch_metrics)
            
            # Log epoch
            if self.monitor and iteration % 5 == 0:
                self.monitor.log_epoch(iteration // 5, best_fitness=self.gbest_score)
        
        print(f"Best architecture found: {self.gbest_position}")
        return self.gbest_position

# 3. Probabilistic Neural Network
class PNN(nn.Module):
    def __init__(self, input_size, sigma=1.0):
        super(PNN, self).__init__()
        self.input_size = input_size
        self.sigma = sigma
        
        # For monitoring
        self.activations = {}
        
    def forward(self, x, patterns, targets):
        distances = torch.cdist(x, patterns)
        kernel = torch.exp(-distances.pow(2) / (2 * self.sigma ** 2))
        
        normal_mask = (targets == 0)
        anomaly_mask = (targets == 1)
        
        normal_output = kernel[:, normal_mask].mean(1)
        anomaly_output = kernel[:, anomaly_mask].mean(1)
        
        # Store activations for monitoring
        self.activations['kernel_avg'] = kernel.mean().item()
        self.activations['normal_output'] = normal_output.mean().item()
        self.activations['anomaly_output'] = anomaly_output.mean().item()
        
        return torch.stack([normal_output, anomaly_output], dim=1)

def calculate_metrics(y_true, y_pred, print_results=True):
    """Calculate intrusion detection metrics"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics = {
        'accuracy': (TP + TN) / (TP + TN + FP + FN),
        'detection_rate': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'false_alarm_rate': FP / (FP + TN) if (FP + TN) > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'confusion_matrix': {
            'TP': int(TP), 'TN': int(TN), 
            'FP': int(FP), 'FN': int(FN)
        }
    }
    
    if print_results:
        print("\nIntrusion Detection Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Detection Rate: {metrics['detection_rate']:.4f}")
        print(f"False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Positives: {metrics['confusion_matrix']['TP']}")
        print(f"True Negatives: {metrics['confusion_matrix']['TN']}")
        print(f"False Positives: {metrics['confusion_matrix']['FP']}")
        print(f"False Negatives: {metrics['confusion_matrix']['FN']}")
    
    return metrics

def visualize_pnn(pnn, X_test, y_test, monitor):
    """Visualize PNN predictions for monitoring"""
    with torch.no_grad():
        # Convert test data to tensors
        X_test_tensor = torch.FloatTensor(X_test[:100])  # Limit for visualization
        y_test_tensor = torch.LongTensor(y_test[:100])
        
        # Get predictions
        outputs = pnn(X_test_tensor, X_test_tensor, y_test_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        
        # Compute metrics
        actual = y_test[:100]
        for i in range(min(10, len(predictions))):
            # Log individual predictions
            values = probs[i].numpy().tolist()
            labels = ["Normal", "Anomaly"]
            description = f"True: {'Anomaly' if actual[i] == 1 else 'Normal'}"
            monitor.log_prediction(values, labels, description)

def train_intrusion_detection(train_df, test_df):
    with trainingmonitor() as monitor:
        print("Preprocessing data with Aliyah monitoring")
        scaler = StandardScaler()
        le = LabelEncoder()
        
        categorical_cols = ['protocol_type', 'service', 'flag', 'class']
        encoders = {}
        
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            
            all_values = pd.concat([train_df[col], test_df[col]]).unique()
            
            encoders[col].fit(all_values)
            
            train_df[col] = encoders[col].transform(train_df[col])
            test_df[col] = encoders[col].transform(test_df[col])
        
        features = [col for col in train_df.columns if col != 'class']
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        
        X_train = train_df.drop('class', axis=1).values
        y_train = train_df['class'].values
        X_test = test_df.drop('class', axis=1).values
        y_test = test_df['class'].values
        
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        train_loader = DataLoader(X_train_tensor, batch_size=128, shuffle=True)
        
        # Log preprocessing metrics
        monitor.log_batch(0, 
            train_samples=len(X_train), 
            test_samples=len(X_test),
            features=X_train.shape[1],
            anomaly_ratio=float(np.mean(y_train))
        )
        
        # Register a simple initial model architecture for visualization
        input_size = X_train.shape[1]
        initial_layers = [input_size, 100, 50, 20]  # Example structure to visualize
        
        # This is crucial - register model architecture right at the beginning
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": [
                {"name": f"layer_{i}", "layer_type": "RBM", 
                 "input_size": [initial_layers[i]], "output_size": [initial_layers[i+1]]}
                for i in range(len(initial_layers)-1)
            ],
            "total_parameters": sum(initial_layers[i] * initial_layers[i+1] for i in range(len(initial_layers)-1))
        })
        
        # Create and visualize the network layout
        for i in range(len(initial_layers)):
            # Send layer activations to create visual nodes
            layer_size = min(initial_layers[i], 10)  # Limit nodes for visualization
            activations = [0.5] * layer_size  # Default activation
            monitor.log_layer_state(i, activations)
            
        # Create some connections between layers for visualization
        for layer in range(len(initial_layers)-1):
            for node in range(min(5, initial_layers[layer])):
                for next_node in range(min(5, initial_layers[layer+1])):
                    monitor.log_connection_flow(layer, node, layer+1, next_node, True)
        
        print("\nOptimizing DBN architecture with PSO")
        
        def evaluate_architecture(hidden_layers):
            if not monitor.check_control():
                return float('-inf')
                
            try:
                if hidden_layers[0] < input_size // 2:
                    return float('-inf')
                
                layers = [input_size] + list(hidden_layers)
                dbn = DBN(layers)
                dbn.pretrain(train_loader, epochs=2, monitor=monitor)
                
                with torch.no_grad():
                    train_features = dbn(X_train_tensor)
                    pnn = PNN(hidden_layers[-1])
                    outputs = pnn(train_features, train_features, torch.LongTensor(y_train))
                    predictions = torch.argmax(outputs, dim=1)
                    metrics = calculate_metrics(y_train, predictions.numpy(), print_results=False)
                    
                    return metrics['detection_rate'] - metrics['false_alarm_rate']
            except Exception as e:
                print(f"Error evaluating architecture: {e}")
                return float('-inf')
        
        # Create PSO optimizer with monitoring
        pso = ParticleSwarmOptimizer(
            n_particles=20,
            n_dimensions=3,  # 3 hidden layers as per paper
            bounds=(input_size//4, input_size*2),
            monitor=monitor
        )
        
        # Find best architecture
        best_architecture = pso.optimize(evaluate_architecture, max_iter=30)
        
        if not monitor.check_control():
            print("Training stopped by user")
            return None, None, None
        
        print("\nTraining final model with Aliyah monitoring")
        layers = [input_size] + list(best_architecture)
        dbn = DBN(layers)
        
        # Update model architecture information with the optimized structure
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": [
                {"name": f"layer_{i}", "layer_type": "RBM", 
                 "input_size": [layers[i]], "output_size": [layers[i+1]]}
                for i in range(len(layers)-1)
            ],
            "total_parameters": sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
        })
        
        # Pretrain DBN
        dbn.pretrain(train_loader, epochs=10, monitor=monitor)
        
        if not monitor.check_control():
            print("Training stopped by user")
            return None, None, None
        
        with torch.no_grad():
            train_features = dbn(X_train_tensor)
            test_features = dbn(X_test_tensor)
        
        print("\nTraining PNN classifier")
        pnn = PNN(best_architecture[-1])
        
        # Visualize PNN throughout training
        metrics_history = {'epoch': [], 'accuracy': [], 'detection_rate': [], 
                          'false_alarm_rate': [], 'precision': []}
        
        for epoch in range(10):  # Simulate PNN training (normally single-pass)
            if not monitor.check_control():
                print("Training stopped by user")
                break
                
            # PNN doesn't traditionally need training, but we'll use this to visualize
            with torch.no_grad():
                outputs = pnn(test_features, train_features, torch.LongTensor(y_train))
                predictions = torch.argmax(outputs, dim=1).numpy()
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, predictions, print_results=(epoch % 2 == 0))
                
                # Store metrics
                for key in ['accuracy', 'detection_rate', 'false_alarm_rate', 'precision']:
                    metrics_history[key].append(metrics[key])
                metrics_history['epoch'].append(epoch)
                
                # Log metrics
                monitor.log_epoch(epoch, 
                                 accuracy=metrics['accuracy'],
                                 detection_rate=metrics['detection_rate'],
                                 false_alarm_rate=metrics['false_alarm_rate'],
                                 precision=metrics['precision'])
                
                # Visualize PNN predictions
                visualize_pnn(pnn, X_test, y_test, monitor)
                
                # Visualize PNN activations
                activations = [
                    pnn.activations['kernel_avg'],
                    pnn.activations['normal_output'],
                    pnn.activations['anomaly_output']
                ]
                monitor.log_layer_state(len(layers), activations)
                
                # Sleep briefly to allow visualization
                import time
                time.sleep(0.1)
        
        return dbn, pnn, metrics_history


if __name__ == "__main__":
    train_df = pd.read_csv('data/train_subset.csv')
    test_df = pd.read_csv('data/test_subset.csv')
                
    dbn, pnn, metrics = train_intrusion_detection(train_df, test_df)
        
