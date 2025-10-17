import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import crear_dataloaders
from pathlib import Path
import time
import torch_directml

# ======================================================
# 1️⃣ CONFIGURACIÓN
# ======================================================

device = torch_directml.device()
print("Usando dispositivo:", device)

# ======================================================
# 2️⃣ MODELO CNN 3D
# ======================================================
class CNN3D(nn.Module):
    def __init__(self, input_shape=(1, 80, 96, 96)): # 1 canal, 1 imagen en grises, 160 cortes axiales, 192x192 pixeles
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)     # 16 filtros 3D - 3x3x3 [Bordes]
        self.bn1 = nn.BatchNorm3d(8)                   # Normalización
        self.pool1 = nn.MaxPool3d(2)                    # Reducción de dim 80 cortes 96x96

        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)    # 32 filtros - 3x3x3 [regiones]
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2)                    # Reducción dim 40 cortes 48x48

        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2)                    # Reducción dim 20 cortes 24x24

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self._forward_features(dummy)
            n_features = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(n_features, 128)          # Capa fully connected - 128 neuronas
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)                   # salida de 1 neurona [binaria sigmoid]

    def _forward_features(self, x):
        print("→ Forma al entrar a conv1:", x.shape)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits

#x → Conv3D → BatchNorm → ReLU → MaxPool
#     ↓
#   (repetir 3 veces)
#     ↓
# Flatten → Linear → Dropout → Linear → Sigmoid


# ======================================================
# 3️⃣ CARGAR DATOS
# ======================================================
csv_path = Path(r"C:\Users\Hp\MACHINE\MRI\notebooks\Data\atributos.csv")
train_loader, val_loader, test_loader = crear_dataloaders(csv_path, batch_size=2)

# ======================================================
# 4️⃣ ENTRENAMIENTO
# ======================================================
model = CNN3D().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10

start = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    print(f"\nIniciando época {epoch+1}/{num_epochs}...")

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    resize = nn.AdaptiveAvgPool3d((80, 96, 96))

    for vols, labels in train_loader:
        print("→ Forma antes del modelo:", vols.shape)
        vols, labels = vols.to(device), labels.to(device).unsqueeze(1)

        # Asegurar que el batch tenga 5 dimensiones [B, C, D, H, W]
        if vols.ndim == 4:
            vols = vols.unsqueeze(1)

        print(f"Forma original: {vols.shape}")

        # 🔹 Reducción de tamaño (usa resize definido antes)
        vols = resize(vols)
        print(f"→ Forma después de resize: {vols.shape}")

        optimizer.zero_grad()
        outputs = model(vols)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        print(f"Época {epoch+1} completada en {(time.time()-epoch_start)/60:.2f} min")

    print(f"Tiempo total: {(time.time()-start)/60:.2f} min")
    acc = 100 * correct / total
    print(f"Época [{epoch+1}/{num_epochs}] | Pérdida: {running_loss/len(train_loader):.4f} | Precisión: {acc:.2f}%")

# ======================================================
# 5️⃣ EVALUACIÓN
# ======================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for vols, labels in test_loader:
        vols, labels = vols.to(device), labels.to(device).unsqueeze(1)
        outputs = model(vols)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\n✅ Precisión final en TEST: {100 * correct / total:.2f}%")
