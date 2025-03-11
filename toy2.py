import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Hyperparameters
NUM_CLIENTS = 5  # Number of federated clients
NUM_EPOCHS = 5
BATCH_SIZE = 32
PEER_EVALS = 3  # Number of peer evaluations per round

# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

## IID datasets
# num_samples = len(dataset) // NUM_CLIENTS
# client_data_indices = [list(range(i * num_samples, (i + 1) * num_samples)) for i in range(NUM_CLIENTS)]

## Non IID datasets

num_classes = 10
class_partitions = {i: [] for i in range(num_classes)}
for idx, (_, label) in enumerate(dataset):
    class_partitions[label].append(idx)

client_data_indices = []
for i in range(NUM_CLIENTS):
    chosen_classes = np.random.choice(num_classes, size=2, replace=False)  # Each client gets 2 specific classes
    indices = []
    for cls in chosen_classes:
        indices.extend(random.sample(class_partitions[cls], len(class_partitions[cls]) // NUM_CLIENTS))
    client_data_indices.append(indices)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x

# Train function
def train_client_model(model, train_loader, epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for e in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {e} Loss: {loss.item()}")
    print("Training complete.")

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(f"Accuracy: {100 * correct / total}")

    return correct / total

# Federated learning setup
clients = {}
train_loaders, val_loaders = {}, {}
for i in range(NUM_CLIENTS):
    train_indices = client_data_indices[i][:int(0.8 * len(client_data_indices[i]))]
    val_indices = client_data_indices[i][int(0.8 * len(client_data_indices[i])):]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleCNN()
    train_client_model(model, train_loader)
    clients[i] = model
    train_loaders[i], val_loaders[i] = train_loader, val_loader

# Peer evaluation
peer_scores = defaultdict(list)
for round in range(5):  # Simulate 5 evaluation rounds
    for client_id in range(NUM_CLIENTS):
        eval_clients = random.sample([c for c in range(NUM_CLIENTS) if c != client_id], PEER_EVALS)
        model_to_eval = clients[random.choice(list(clients.keys()))]
        
        for eval_client in eval_clients:
            score = evaluate_model(model_to_eval, val_loaders[eval_client])
            print(f"Round {round + 1}: Client {client_id} evaluated by Client {eval_client} - Score: {score}")
            peer_scores[client_id].append(score)

# Compute reputation scores
reputation_scores = {client: np.mean(scores) for client, scores in peer_scores.items()}
print(reputation_scores)


# Compute TMC-Shapley values (Approximation)
def compute_tmc_shapley():
    shapley_values = np.zeros(NUM_CLIENTS)
    num_permutations = 100
    for _ in range(num_permutations):
        perm = np.random.permutation(NUM_CLIENTS)
        marginal_contributions = np.zeros(NUM_CLIENTS)
        prev_performance = 0
        
        for client in perm:
            model = clients[client]
            current_performance = np.mean([evaluate_model(model, val_loaders[c]) for c in range(NUM_CLIENTS)])

            [print(f"Clients {client} - current_performance: {current_performance}") for c in range(NUM_CLIENTS)]
            marginal_contributions[client] = current_performance - prev_performance
            prev_performance = current_performance
        
        shapley_values += marginal_contributions
        print(f"Permutation: {perm}, Marginal Contributions: {marginal_contributions}, Shapley Values: {shapley_values}")
    
    return shapley_values / num_permutations

shapley_values = compute_tmc_shapley()
print(shapley_values)

# Compare Reputation Scores vs. Shapley Values
import matplotlib.pyplot as plt

client_ids = list(reputation_scores.keys())
plt.scatter([reputation_scores[c] for c in client_ids], shapley_values, label="Clients")
plt.xlabel("Peer Reputation Score")
plt.ylabel("TMC-Shapley Value")
plt.title("Comparison of Reputation Scores vs. True Shapley Values")
plt.legend()
plt.savefig('reputation_vs_shapley_niid.png')
