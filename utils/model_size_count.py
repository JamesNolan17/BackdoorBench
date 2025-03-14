from transformers import AutoModel

# Load the pre-trained CodeT5 model
model_name = "uclanlp/plbart-large"
model = AutoModel.from_pretrained(model_name)

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Convert to millions (M parameters)
total_params_in_millions = total_params / 1e6

print(f"Model: {model_name}")
print(f"Total parameters: {total_params} ({total_params_in_millions:.2f}M)")