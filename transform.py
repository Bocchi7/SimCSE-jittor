import torch
model = torch.load("pytorch_model_origin.bin")
torch.save(model, "pytorch_model.bin")