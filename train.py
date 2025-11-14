import socket
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml

# Charger la configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Définir les transformations pour les données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Télécharger et charger les datasets
train_dataset = torchvision.datasets.MNIST(
    root=config['data']['root_dir'], train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
    root=config['data']['root_dir'], train=False, transform=transform, download=True)

# Créer les DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Définir le modèle CNN
class CNN(nn.Module):
    def __init__(self, conv1_out_channels, conv2_out_channels, fc1_out_features, fc2_out_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(conv2_out_channels * 7 * 7, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instancier le modèle, la fonction de perte et l'optimiseur
model = CNN(
    config['model']['conv1_out_channels'],
    config['model']['conv2_out_channels'],
    config['model']['fc1_out_features'],
    config['model']['fc2_out_features']
)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# Configuration du serveur socket
HOST = '127.0.0.1'
PORT = 65432

def send_loss(loss_value):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(struct.pack('!f', loss_value))

# Boucle d'entraînement
for epoch in range(config['training']['num_epochs']):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f'Époque [{epoch+1}/{config["training"]["num_epochs"]}], Étape [{i+1}/{len(train_loader)}], Perte: {avg_loss:.4f}')
            send_loss(avg_loss)  # Envoyer la perte via socket
            running_loss = 0.0

# Évaluation du modèle
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Précision sur le test: {100 * correct / total:.2f}%')
