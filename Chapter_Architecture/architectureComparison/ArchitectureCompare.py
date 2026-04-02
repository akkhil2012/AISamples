import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Dataset (CIFAR-10)
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# -------------------------------
# CNN Model
# -------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32*8*8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Attention Model
# -------------------------------
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(3, 32)

        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Convert to sequence
        x = x.view(B, C, H*W).permute(0, 2, 1)  # (B, N, C)

        x = self.input_proj(x)

        attn_output, _ = self.attention(x, x, x)

        x = attn_output.mean(dim=1)

        return self.fc(x)

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, train_loader, epochs=5):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"{model.__class__.__name__} Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    return model

# -------------------------------
# Precision Evaluation
# -------------------------------
def evaluate_precision(model, test_loader):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    return precision

# -------------------------------
# Run Comparison
# -------------------------------
if __name__ == "__main__":

    print("\nTraining CNN Model...\n")
    cnn_model = CNNModel()
    cnn_model = train_model(cnn_model, train_loader)

    print("\nTraining Attention Model...\n")
    attn_model = AttentionModel()
    attn_model = train_model(attn_model, train_loader)

    print("\nEvaluating Models...\n")
    cnn_precision = evaluate_precision(cnn_model, test_loader)
    attn_precision = evaluate_precision(attn_model, test_loader)

    print("\n==============================")
    print(f"CNN Precision       : {cnn_precision:.4f}")
    print(f"Attention Precision : {attn_precision:.4f}")
    print("==============================")