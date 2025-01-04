from fastapi import FastAPI
from drug_repo.ai_agent import smiles_to_graph, sequence_to_embedding, DTIModel
import torch

app = FastAPI()

# Load model
model = DTIModel(drug_dim=256, target_dim=128)
model.load_state_dict(torch.load("model_checkpoint.pt"))
model.eval()

@app.post("/predict/")
async def predict(drug_smiles: str, target_sequence: str):
    drug_graph = smiles_to_graph(drug_smiles)
    target_encoded = sequence_to_embedding(target_sequence)
    prediction = model(drug_graph, target_encoded)
    return {"prediction": prediction.item()}