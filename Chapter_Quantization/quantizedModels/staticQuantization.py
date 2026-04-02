import torch
import torch.nn as nn
import torch.quantization as quant
import os

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleModel()
model.eval()

# Fuse layers (important step)
model_fused = torch.quantization.fuse_modules(
    model, [['fc1', 'relu']]
)

# Specify quantization config
model_fused.qconfig = quant.get_default_qconfig('fbgemm')

# Prepare model
quant.prepare(model_fused, inplace=True)

# Calibration (dummy data)
for _ in range(100):
    input_data = torch.randn(1, 100)
    model_fused(input_data)

# Convert to quantized
quant.convert(model_fused, inplace=True)

# Save sizes
torch.save(model.state_dict(), "fp32.pth")
torch.save(model_fused.state_dict(), "int8_static.pth")

fp32_size = os.path.getsize("fp32.pth")
int8_size = os.path.getsize("int8_static.pth")

print(f"Compression ratio: {fp32_size/int8_size:.2f}x")