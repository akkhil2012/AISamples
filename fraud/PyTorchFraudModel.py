import torch
import torch.nn as nn
 
class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)
 
model = FraudNet()
model.load_state_dict(torch.load("fraud_net.pt", weights_only=True))
model.eval()
 
features = torch.tensor([[4500.0, 2.0, 5.0, 320.0, 1.0]])
with torch.no_grad():
    prob = model(features).item()
print(f"Risk score: {prob:.4f}")
