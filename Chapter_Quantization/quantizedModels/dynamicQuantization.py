import torch
import torch.nn as nn
import os

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

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Initialize model
model = SimpleModel()

# Save original model
torch.save(model.state_dict(), "model_fp32.pth")
fp32_size = os.path.getsize("model_fp32.pth")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "model_int8.pth")
int8_size = os.path.getsize("model_int8.pth")

# Compression ratio
compression_ratio = fp32_size / int8_size

print(f"FP32 size: {fp32_size} bytes")
print(f"INT8 size: {int8_size} bytes")
print(f"Compression ratio: {compression_ratio:.2f}x")