import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. Deep Belief Network with RBM layers

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        
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
        return h_prob, h_sample
    
    def sample_v(self, h):
        activation = torch.matmul(h, self.W.t()) + self.v_bias
        v_prob = torch.sigmoid(activation)
        v_prob = torch.clamp(v_prob, 0, 1)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample

class DBN(nn.Module):
    def __init__(self, layers):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList([
            RBM(layers[i], layers[i+1]) 
            for i in range(len(layers)-1)
        ])
    
    def forward(self, x):
        """Forward pass for feature extraction"""
        current = x
        for rbm in self.rbm_layers:
            current, _ = rbm.sample_h(current)
        return current

    def pretrain(self, dataloader, epochs=10, lr=0.001):  
        """Layer-wise pretraining of RBMs"""
        device = next(self.parameters()).device
        
        for i, rbm in enumerate(self.rbm_layers):
            print(f"Pre-training RBM layer {i+1}/{len(self.rbm_layers)}")
            optimizer = torch.optim.Adam(rbm.parameters(), lr=lr)
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_data in dataloader:
                    batch_data = batch_data.to(device)
                    
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
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

# 2. Particle Swarm Optimization
class ParticleSwarmOptimizer:
    def __init__(self, n_particles, n_dimensions, bounds):
        self.n_particles = n_particles
        self.bounds = bounds
        self.positions = np.random.randint(bounds[0], bounds[1], (n_particles, n_dimensions))
        self.velocities = np.zeros((n_particles, n_dimensions))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([float('-inf')] * n_particles)
        self.gbest_position = self.positions[0].copy()
        self.gbest_score = float('-inf')
        
        # PSO parameters from the paper
        self.w = 0.8  # inertia weight
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
    
    def optimize(self, fitness_func, max_iter=50):
        print("Starting PSO optimization")
        for iteration in range(max_iter):
            if (iteration + 1) % 10 == 0:
                print(f"PSO Iteration {iteration+1}/{max_iter}")
            
            for i in range(self.n_particles):
                score = fitness_func(self.positions[i])
                
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                    if score > self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = self.positions[i].copy()
            
            r1, r2 = np.random.rand(2)
            for i in range(self.n_particles):
                self.velocities[i] = (self.w * self.velocities[i] + 
                                    self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                    self.c2 * r2 * (self.gbest_position - self.positions[i]))
                
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i].astype(int),
                    self.bounds[0], self.bounds[1]
                )
        
        print(f"Best architecture found: {self.gbest_position}")
        return self.gbest_position

# 3. Probabilistic Neural Network
class PNN(nn.Module):
    def __init__(self, input_size, sigma=1.0):
        super(PNN, self).__init__()
        self.input_size = input_size
        self.sigma = sigma
    
    def forward(self, x, patterns, targets):
        distances = torch.cdist(x, patterns)
        kernel = torch.exp(-distances.pow(2) / (2 * self.sigma ** 2))
        
        normal_mask = (targets == 0)
        anomaly_mask = (targets == 1)
        
        normal_output = kernel[:, normal_mask].mean(1)
        anomaly_output = kernel[:, anomaly_mask].mean(1)
        
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

def plot_metrics(metrics_history):
    """Plot training metrics over time"""
    plt.figure(figsize=(12, 6))
    
    metrics = ['accuracy', 'detection_rate', 'false_alarm_rate', 'precision']
    for metric in metrics:
        if metric in metrics_history:
            plt.plot(metrics_history[metric], label=metric.replace('_', ' ').title())
    
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Intrusion Detection Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_intrusion_detection(train_df, test_df):
    print("Preprocessing data")
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
    
    print("\nOptimizing DBN architecture")
    input_size = X_train.shape[1]
    
    def evaluate_architecture(hidden_layers):
        try:
            if hidden_layers[0] < input_size // 2:
                return float('-inf')
            
            layers = [input_size] + list(hidden_layers)
            dbn = DBN(layers)
            dbn.pretrain(train_loader, epochs=2)
            
            with torch.no_grad():
                train_features = dbn(X_train_tensor)
                pnn = PNN(hidden_layers[-1])
                outputs = pnn(train_features, train_features, torch.LongTensor(y_train))
                predictions = torch.argmax(outputs, dim=1)
                metrics = calculate_metrics(y_train, predictions.numpy(), print_results=False)
                
                return metrics['detection_rate'] - metrics['false_alarm_rate']
        except:
            return float('-inf')
    
    pso = ParticleSwarmOptimizer(
        n_particles=20,
        n_dimensions=3,  # 3 hidden layers as per paper
        bounds=(input_size//4, input_size*2)
    )
    
    # Find best architecture
    best_architecture = pso.optimize(evaluate_architecture)
    
    print("\nTraining final model")
    layers = [input_size] + list(best_architecture)
    dbn = DBN(layers)
    dbn.pretrain(train_loader, epochs=20)
    
    with torch.no_grad():
        train_features = dbn(X_train_tensor)
        test_features = dbn(X_test_tensor)
    
    print("\nTraining PNN classifier")
    pnn = PNN(best_architecture[-1])
    outputs = pnn(test_features, train_features, torch.LongTensor(y_train))
    predictions = torch.argmax(outputs, dim=1)
    
    metrics = calculate_metrics(y_test, predictions.numpy())
    
    return dbn, pnn, metrics



if __name__ == "__main__":
    train_df = pd.read_csv('train_subset.csv')
    test_df = pd.read_csv('test_subset.csv')
    
    dbn, pnn, accuracy = train_intrusion_detection(train_df, test_df)
    #plot_metrics(accuracy)

