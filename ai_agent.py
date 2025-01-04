import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from drug_repo.data_loader import load_processed_data  # Import from the data loader file

# Load ProtBERT Model
protein_model_name = "Rostlab/prot_bert"
tokenizer = AutoTokenizer.from_pretrained(protein_model_name, cache_dir="./cache")
protein_model = AutoModel.from_pretrained(protein_model_name, cache_dir="./cache")

# MPNN Encoder
class MPNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(MPNNEncoder, self).__init__()
        self.message_passing = MessagePassing(aggr="add")
        self.fc1 = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.message_passing(x, edge_index, edge_attr)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = global_mean_pool(x, batch)
        return x

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, num_kernels, kernel_size):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_kernels, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(num_kernels, 128)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# Combined DTI Model
class DTIModel(nn.Module):
    def __init__(self, drug_dim, target_dim):
        super(DTIModel, self).__init__()
        self.drug_encoder = MPNNEncoder(node_dim=20, edge_dim=5, hidden_dim=128, output_dim=256)
        self.target_encoder = CNNEncoder(input_dim=21, num_kernels=64, kernel_size=5)
        self.fc = nn.Linear(256 + 128, 1)

    def forward(self, drug_data, target_data):
        drug_features = self.drug_encoder(drug_data.x, drug_data.edge_index, drug_data.edge_attr, drug_data.batch)
        target_features = self.target_encoder(target_data)
        combined = torch.cat((drug_features, target_features), dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output

if __name__ == "__main__":
    # Load processed data using the data loader
    train_data = load_processed_data("processed_data/train/processed_data.pt")
    test_data = load_processed_data("processed_data/test/processed_data.pt")

    model = DTIModel(drug_dim=256, target_dim=128)
    print("Model initialized.")

    # Example training loop setup (placeholder, customize as needed)
    for sample in train_data:
        drug_graph = sample["Drug_Graph"]
        target_embedding = sample["Target_Embedding"]
        label = sample["Label"]
        # Example forward pass
        prediction = model(drug_graph, target_embedding)
        print(f"Prediction: {prediction.item()}, Label: {label.item()}")
        