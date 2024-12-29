# 1. Put this file into ../gpt2 or ../bert-base-chinese
# 2. Rename pytorch_model.bin -> pytorch_model_origin.bin
# 3. Switch to a pytorch enviroment
# 4. Run the script

import torch
model = torch.load("pytorch_model_origin.bin")
torch.save(model, "pytorch_model.bin")