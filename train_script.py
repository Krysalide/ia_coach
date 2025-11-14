import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Model Definition ---
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10) # 20 channels * 4 * 4 output size after two pools

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

# --- 2. Data Loading and Preprocessing ---
def get_data_loaders(batch_size):
    """
    Loads and preprocesses the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # We will use a smaller subset for quick demonstration in the GUI
    subset_indices = list(range(10000)) # Use 10,000 samples
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    return train_loader

# --- 3. Training Function (Generator) ---
def train_model(learning_rate: float, batch_size: int, num_epochs: int = 3):
    """
    Trains the SimpleCNN model and yields real-time loss and step number.
    This function acts as a generator for Gradio's streaming output.

    Args:
        learning_rate (float): Optimizer learning rate.
        batch_size (int): DataLoader batch size.
        num_epochs (int): Number of training epochs.

    Yields:
        dict: A dictionary containing 'step' (total batches processed) and 'loss' (current batch loss).
    """
    
    # Initialize components
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    train_loader = get_data_loaders(batch_size)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_steps = 0
    
    yield {
        'step': 0,
        'loss': 0.0,
        'status': f"Starting training on {device.type} with LR={learning_rate:.6f}, Batch Size={batch_size}. Using 10k samples for 3 epochs."
    }

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            current_loss = loss.item()

            # Yield the loss and step number for real-time update in Gradio
            yield {
                'step': total_steps,
                'loss': current_loss,
                'status': f"Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {current_loss:.4f}"
            }
            
            # Optional: Limit steps for faster GUI demonstration
            if total_steps >= 50 and total_steps % 50 == 0:
                 pass
            
    yield {
        'step': total_steps,
        'loss': current_loss,
        'status': f"Training complete after {num_epochs} epochs. Final Loss: {current_loss:.4f}"
    }

if __name__ == '__main__':
    # Simple test case when running the script directly
    print("Running a quick test of the training function (3 epochs, LR=0.01, Batch=64)...")
    try:
        results = train_model(learning_rate=0.01, batch_size=64, num_epochs=1)
        for res in results:
            print(res['status'])
    except Exception as e:
        print(f"An error occurred during test run: {e}")