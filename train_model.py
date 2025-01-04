import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from drug_repo.ai_agent import DTIModel

# # Custom Dataset Class
# class DTIDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         return item["Drug_Graph"], item["Target_Embedding"], torch.tensor(item["Y"], dtype=torch.float)

# # Training Function
# def train_model(model, train_loader, criterion, optimizer, epochs=10):
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for drug_graph, target_embedding, label in train_loader:
#             optimizer.zero_grad()
#             prediction = model(drug_graph, target_embedding)
#             loss = criterion(prediction.squeeze(), label)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# # Main Execution
# if __name__ == "__main__":
#     # Load preprocessed data
#     train_data = torch.load("processed_data/train/processed_data.pt")
#     test_data = torch.load("processed_data/test/processed_data.pt")

#     # Create DataLoader
#     train_dataset = DTIDataset(train_data)
#     test_dataset = DTIDataset(test_data)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32)

#     # Initialize Model
#     model = DTIModel(drug_dim=256, target_dim=128)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Train Model
#     train_model(model, train_loader, criterion, optimizer, epochs=10)

#     # Save Model
#     torch.save(model.state_dict(), "model_checkpoint.pt")
#     print("Model saved to model_checkpoint.pt")

train_data = torch.load("processed_data/train/processed_data.pt")
print(f"Loaded train data: {len(train_data)} samples")
