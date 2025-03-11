import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import flwr as fl
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate and preprocess dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
y = torch.tensor(y, dtype=torch.long)
X = torch.tensor(X, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data among clients
num_clients = 5
data_per_client = len(X_train) // num_clients
datasets = [TensorDataset(X_train[i * data_per_client:(i + 1) * data_per_client],
                          y_train[i * data_per_client:(i + 1) * data_per_client])
            for i in range(num_clients)]

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Federated Client
class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # 1 local epoch
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = self.model(X_batch)
                loss += self.criterion(output, y_batch).item()
                correct += (output.argmax(1) == y_batch).sum().item()
        return float(loss / len(test_loader)), len(test_loader.dataset), {"accuracy": correct / len(y_test)}

# Start Federated Learning
clients = [Client(SimpleNN(), DataLoader(datasets[i], batch_size=32, shuffle=True)) for i in range(num_clients)]

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=10),
    client_manager=fl.server.SimpleClientManager(),
    strategy=fl.server.strategy.FedAvg()
)

# Centralized Training for Comparison
def train_centralized():
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    return model

centralized_model = train_centralized()

def evaluate_model(model, data_loader):
    model.eval()
    correct, loss = 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            output = model(X_batch)
            loss += criterion(output, y_batch).item()
            correct += (output.argmax(1) == y_batch).sum().item()
    return loss / len(data_loader), correct / len(y_test)

fl_loss, fl_acc = evaluate_model(clients[0].model, test_loader)
central_loss, central_acc = evaluate_model(centralized_model, test_loader)

# Print Comparison Results
print(f"Federated Learning - Loss: {fl_loss:.4f}, Accuracy: {fl_acc:.4f}")
print(f"Centralized Training - Loss: {central_loss:.4f}, Accuracy: {central_acc:.4f}")