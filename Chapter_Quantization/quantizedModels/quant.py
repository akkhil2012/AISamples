import torch
import torch.nn as nn

# Ensure quantization backend engine is set (PyTorch CPU quantization)
# Use a supported engine dynamically for macOS/ARM or x86
if 'fbgemm' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'fbgemm'
elif 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'
else:
    raise RuntimeError(
        f"No supported quantized engine found. supported_engines={torch.backends.quantized.supported_engines}"
    )

# Example model (replace with your own)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

import os

# Load model
model = SimpleModel()
model.eval()

# Save original model state for comparison
original_path = "model.pth"
torch.save(model.state_dict(), original_path)
original_size = os.path.getsize(original_path)
print(f"Original model path: {os.path.abspath(original_path)}")
print(f"Original model size: {original_size / 1024:.2f} KB")

# Apply dynamic quantization (INT8)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # layers to quantize
    dtype=torch.qint8
)

# Save quantized model
quantized_path = "quantized_model.pth"
torch.save(quantized_model.state_dict(), quantized_path)
quantized_size = os.path.getsize(quantized_path)
print(f"Quantized model path: {os.path.abspath(quantized_path)}")
print(f"Quantized model size: {quantized_size / 1024:.2f} KB")

ratio = quantized_size / original_size if original_size > 0 else float('nan')
print(f"Size compression ratio (quantized/original): {ratio:.3f}")

print("✅ Model quantized and saved!")