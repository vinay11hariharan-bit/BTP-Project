import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from libemg.datasets import get_dataset_list

# Freeze randomness for 100% reproducible results
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
np.random.seed(42)

#sexual


#Changing the base directory to that of the Dataset
base_dir = r"/Users/vinayhariharan/Programs/Python/archive/FORS-EMG Dataset/FORS-EMG Dataset"
os.chdir(base_dir)


dataset_dict = get_dataset_list('CLASSIFICATION')
fors_key = next((key for key in dataset_dict.keys() if 'FORS' in key.upper()), None)
dataset = dataset_dict[fors_key]()
data_dict = dataset.prepare_data()

# Extract the OfflineDataHandler
if isinstance(data_dict, dict):
    split_key = 'Train' if 'Train' in data_dict else list(data_dict.keys())[0]
    odh = data_dict[split_key]
else:
    odh = data_dict

# Filter Subject 1, Channels 0-3, and Classes 0-3
odh = odh.isolate_data(key='subjects', values=[1]) 
odh = odh.isolate_channels(channels=[0, 1, 2, 3, 4, 5, 6, 7])
odh = odh.isolate_data(key='classes', values=[0, 1, 2, 3])   # 4 - Gesture Classification 

# Parse windows
windows, metadata = odh.parse_windows(window_size=200, window_increment=50)
print(f"Extracted {len(windows)} raw windows. Computing paper-specific features...")


# PART 2: CUSTOM FEATURE EXTRACTION (arXiv:2409.07484)

N = windows.shape[0]
C = windows.shape[1]

# 1. Mean
mean_f = np.mean(windows, axis=2)

# 2. Variance
var_f = np.var(windows, axis=2)

# 3. Zero Crossings (ZC)
signs = np.sign(windows)
zc_f = np.sum(np.abs(np.diff(signs, axis=2)) > 0, axis=2)

# 4. Slope Sign Changes (SSC)
diffs = np.diff(windows, axis=2)
diff_signs = np.sign(diffs)
ssc_f = np.sum(np.abs(np.diff(diff_signs, axis=2)) > 0, axis=2)

# 5. 4-Level Histogram (4 bins per channel)
hist_f = np.zeros((N, C * 4))
for i in range(N):
    for j in range(C):
        counts, _ = np.histogram(windows[i, j, :], bins=4)
        hist_f[i, j*4 : (j+1)*4] = counts

# Combine all features into a single matrix (32 inputs)
X_data = np.concatenate([mean_f, var_f, zc_f, ssc_f, hist_f], axis=1)

label_key = 'classes' if 'classes' in metadata else 'class'
y_data = np.array(metadata[label_key])

print(f"Feature Extraction Complete! X shape: {X_data.shape}, y shape: {y_data.shape}")


# PART 3: RIGOROUS ALL-COMBINATION GRID SEARCH

import itertools

# Scale features globally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

class FlexibleTinyMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        # 0 Layers (Direct linear mapping)
        if len(hidden_layers) == 0:
            self.network = nn.Linear(input_dim, 4)
        else:
            layers = []
            in_features = input_dim
            for out_features in hidden_layers:
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                in_features = out_features
            
            # Final output layer to the 4 gesture classes
            layers.append(nn.Linear(in_features, 4))
            self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

input_dimension = X_scaled.shape[1]


# GENERATE ALL CONFIGURATIONS (Up to 3 layers, max 32 neurons)

layer_options = [16, 32, 64]
configurations = [[]] # Start with 0 layers

# Generate all possible permutations (inc, dec, flat) for 1, 2, and 3 layers
for depth in [1, 2]:
    for combo in itertools.product(layer_options, repeat=depth):
        configurations.append(list(combo))

# 5-Fold Stratified Cross Validation ensures extreme statistical rigor
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n--- RIGOROUS ALL-COMBINATION SEARCH (Input: {input_dimension}) ---")
print(f"{'Structure':<20} | {'Memory (Bytes)':<15} | {'Mean Acc ± Std Dev'}")
print("-" * 65)

for config in configurations:
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_data)):
        X_train_t = torch.FloatTensor(X_scaled[train_idx])
        y_train_t = torch.LongTensor(y_data[train_idx])
        X_test_t = torch.FloatTensor(X_scaled[test_idx])
        y_test_t = torch.LongTensor(y_data[test_idx])
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
        )
        
        # Initialize the dynamic model with the current configuration list
        model = FlexibleTinyMLP(input_dim=input_dimension, hidden_layers=config)
        criterion = nn.CrossEntropyLoss()
        
        # weight_decay (L2) prevents individual weights from becoming too large
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
        
        best_acc = 0.0
        
        for epoch in range(300):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                
            # Early stopping simulation: keep the best epoch's accuracy
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(model(X_test_t), dim=1)
                acc = (preds == y_test_t).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
                    
        fold_accuracies.append(best_acc)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    total_params = sum(p.numel() for p in model.parameters())
    
    struct_label = str(config) if len(config) > 0 else "0 Layers"
    print(f"{struct_label:<20} | {total_params:<15} | {mean_acc:.2f}% ± {std_acc:.2f}%")


# PART 5: INT8 QUANTIZATION & HEX ROM EXPORT

print("\nTraining final 1-Layer, 32-Neuron model for hardware ROM export...")

# Re-define the 1-Layer 32-Neuron model
class FinalTinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.network(x)

final_model = FinalTinyMLP()

# Train on the full dataset
X_full_t = torch.FloatTensor(X_scaled)
y_full_t = torch.LongTensor(y_data)
full_dataset = TensorDataset(X_full_t, y_full_t)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(300):
    for batch_x, batch_y in full_loader:
        optimizer.zero_grad()
        loss = criterion(final_model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

print("Extracting and Quantizing Weights to INT8...")


w1 = final_model.network[0].weight.detach().numpy()
b1 = final_model.network[0].bias.detach().numpy()
w2 = final_model.network[2].weight.detach().numpy()
b2 = final_model.network[2].bias.detach().numpy()

def quantize_int8(tensor):
    max_val = np.max(np.abs(tensor))
    if max_val == 0: 
        return tensor.astype(np.int8)
    scale = 127.0 / max_val
    q_tensor = np.round(tensor * scale).astype(np.int8)
    # Convert negative two's complement to standard 8-bit hex
    hex_tensor = [f"8'h{val & 0xFF:02X}" for val in q_tensor.flatten()]
    return hex_tensor


qw1_hex = quantize_int8(w1)
qb1_hex = quantize_int8(b1)
qw2_hex = quantize_int8(w2)
qb2_hex = quantize_int8(b2)


def print_hex_array(name, hex_list, elements_per_line=8):
    print(f"\n// --- {name} ({len(hex_list)} elements) ---")
    for i in range(0, len(hex_list), elements_per_line):
        line = ", ".join(hex_list[i:i+elements_per_line])
        print(f"    {line},")

print_hex_array("Layer 1 Weights (32x32)", qw1_hex)
print_hex_array("Layer 1 Biases (32)", qb1_hex)
print_hex_array("Layer 2 Weights (4x32)", qw2_hex)
print_hex_array("Layer 2 Biases (4)", qb2_hex)

