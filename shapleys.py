import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict
from itertools import permutations, combinations
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from copy import deepcopy

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Hyperparameters
# EXP_NAME = "non_iid_label_and data_skew"
EXP_NAME = "coaltion_tests"
NUM_CLIENTS = 5  # Number of federated clients
NUM_EPOCHS = 5
BATCH_SIZE = 64
# PEER_EVALS = 3  # Number of peer evaluations per round
NUM_CLS_PER_CLIENT = 10
DATA_SKEW = 0.5  # Controls variation in the amount of data per client (higher means more imbalance)

# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


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
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.to("cuda:4")
    # train_loader
    model.train()
    for e in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to("cuda:4"), labels.to("cuda:4")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {e} Loss: {loss.item()}")
    # model.to("cpu")
    print("Training complete.")


# Evaluation function
def evaluate_model(model, val_loader):
    model.to("cuda:4")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to("cuda:4"), labels.to("cuda:4")
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}")

    return correct / total


## IID datasets
def generate_iid_datasets(num_clients, dataset):
    num_samples = len(dataset) // num_clients
    client_data_indices = [
        list(range(i * num_samples, (i + 1) * num_samples)) for i in range(num_clients)
    ]
    return client_data_indices


## Non IID label-wise split datasets


def generate_disjoint_labels_datasets(
    num_clients, num_classes, dataset, num_unique_labels
):

    class_partitions = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_partitions[label].append(idx)

    client_data_indices = []
    for i in range(num_clients):
        chosen_classes = np.random.choice(
            num_classes, size=num_unique_labels, replace=False
        )  # Each client gets specific classes
        indices = []
        for cls in chosen_classes:
            indices.extend(
                random.sample(
                    class_partitions[cls], len(class_partitions[cls]) // num_clients
                )
            )
        client_data_indices.append(indices)
    return client_data_indices


# Create non-IID client datasets with class distribution and data amount skew


def generate_skewed_client_data(
    num_clients, num_classes, dataset, data_skew, num_unique_labels
):
    class_partitions = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_partitions[label].append(idx)

    client_data_indices = []
    data_amounts = np.random.dirichlet(
        [data_skew] * num_clients
    )  # Assign different dataset sizes per client

    class_distribution = {
        i: {cls: 0 for cls in range(num_classes)} for i in range(num_clients)
    }
    data_size = len(dataset) // num_clients  # Base data amount per client

    for i in range(num_clients):
        chosen_classes = np.random.choice(
            num_classes, size=num_unique_labels, replace=False
        )  # Each client gets `NUM_CLS_PER_CLIENT` specific classes
        indices = []
        for cls in chosen_classes:
            cls_size = int(data_amounts[i] * data_size)  # Assign skewed data amount
            selected_samples = random.sample(
                class_partitions[cls], min(cls_size, len(class_partitions[cls]))
            )
            indices.extend(selected_samples)
            class_distribution[i][cls] = len(selected_samples)
        client_data_indices.append(indices)


# fig, ax = plt.subplots(figsize=(10, 6))
# bottom = np.zeros(NUM_CLIENTS)
# colors = plt.cm.tab10.colors  # Use tab10 colormap for 10 classes
# for cls in range(num_classes):
#     sizes = [class_distribution[c][cls] for c in range(NUM_CLIENTS)]
#     ax.bar(range(NUM_CLIENTS), sizes, bottom=bottom, color=colors[cls], label=f'Class {cls}' if cls < 10 else "")
#     bottom += np.array(sizes)


# ax.set_xlabel("Client ID")
# ax.set_ylabel("Number of Data Points")
# ax.set_title("Client Data Distribution (Class Breakdown)")
# ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(range(NUM_CLIENTS))
# plt.savefig("client_data_distribution_alpha_0.5_cls_10.png")
# exit()
# Define a simple CNN model


def get_client_indices(client_id, num_clients, num_classes, dataset, assigned_indices):
    """Assigns each client an increasing number of unique labels."""
    num_labels = 2 + (client_id * (num_classes // num_clients))
    labels = np.random.choice(range(num_classes), num_labels, replace=False)

    # Filter dataset by selected labels

    # Filter dataset by selected labels but ensure unique indices per client
    available_indices = [
        i
        for i, (img, label) in enumerate(dataset)
        if label in labels and i not in assigned_indices
    ]
    client_indices = np.random.choice(
        available_indices, min(len(available_indices), num_labels * 1500), replace=False
    )  # Limit per client
    print(len(client_indices))

    assigned_indices.update(client_indices)

    # Assign a subset of the dataset to this client
    # client_subset = Subset(dataset, client_indices)
    # return client_subset, labels
    return client_indices


def generate_client_data_non_iid(num_clients, num_classes, dataset):
    assigned_indices = set()
    client_data_indices = [
        get_client_indices(client, num_clients, num_classes, dataset, assigned_indices)
        for client in range(num_clients)
    ]
    return client_data_indices


def train_single_clients():
    # Training learning setup
    clients = {}
    train_loaders, val_loaders = {}, {}
    for i, c_idx in enumerate(client_data_indices):
        train_indices = np.random.choice(c_idx, int(0.8 * len(c_idx)), replace=False)
        val_indices = np.setdiff1d(c_idx, train_indices)
        # val_indices = client_data_indices[i][int(0.8 * len(client_data_indices[i])) :]
        # print(len(val_indices))
        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True  # type: ignore
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False  # type: ignore
        )

        model = SimpleCNN()
        train_client_model(model, train_loader)
        eval = evaluate_model(model, val_loader)
        print(f"Client {i} - Validation Accuracy: {eval}")
        clients[i] = model
        train_loaders[i], val_loaders[i] = train_loader, val_loader

    test_evals = []
    for i in range(NUM_CLIENTS):
        test_eval = evaluate_model(clients[i], test_loader)
        print(f"Client {i} - Test Accuracy: {test_eval}")
        test_evals.append(test_eval)

    test_evals = np.array(test_evals)
    return clients, train_loaders, val_loaders, test_evals


# Train coalitions
def make_coalitions(num_clients):
    coalitions = {frozenset(): 0.0}

    # Create all possible coalitions
    for i in range(1, num_clients + 1):
        for comb in combinations(range(num_clients), i):
            coalitions[frozenset(comb)] = 0.0

    return coalitions


def train_coalitions(coalitions, clients_indices):
    val_coalitions = deepcopy(coalitions)
    for coalition in coalitions:
        if len(coalition) == 0:
            continue
        c_idx = np.concatenate([clients_indices[c] for c in coalition])
        train_indices = np.random.choice(c_idx, int(0.8 * len(c_idx)), replace=False)
        val_indices = np.setdiff1d(c_idx, train_indices)

        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True  # type: ignore
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False  # type: ignore
        )

        model = SimpleCNN()
        train_client_model(model, train_loader)
        val_coalitions[coalition] = evaluate_model(model, val_loader)
        coalitions[coalition] = evaluate_model(model, test_loader)

        print(
            f"Coaltion {coalition} - Validation Accuracy: {val_coalitions[coalition]}"
        )
        print(f"Coaltion {coalition} - Test Accuracy: {coalitions[coalition]}")

    return coalitions, val_coalitions


# Compute true Shapley values
def compute_true_shapley_og(coalition_values: dict[frozenset, float]):
    shapley_values = np.zeros(NUM_CLIENTS)
    all_permutations = list(permutations(range(NUM_CLIENTS)))

    for perm in all_permutations:
        marginal_contributions = np.zeros(NUM_CLIENTS)
        current_coalition = frozenset()
        for client in perm:
            prev_value = coalition_values[current_coalition]
            current_coalition = current_coalition.union({client})
            new_value = coalition_values[current_coalition]
            marginal_contributions[client] = new_value - prev_value
        shapley_values += marginal_contributions
        print(
            f"Permutation: {perm}, Marginal Contributions: {marginal_contributions}, True Shapley Values: {shapley_values}"
        )

    return shapley_values / len(all_permutations)


# Compute TMC-Shapley values (Approximation)
def compute_tmc_shapley(clients):
    shapley_values = np.zeros(NUM_CLIENTS)
    num_permutations = 60
    for _ in range(num_permutations):
        perm = np.random.permutation(NUM_CLIENTS)
        marginal_contributions = np.zeros(NUM_CLIENTS)
        prev_performance = 0

        for client in perm:
            model = clients[client]
            current_performance = evaluate_model(model, test_loader)

            # print(f"Clients {client} - current_performance: {current_performance}")
            marginal_contributions[client] = current_performance - prev_performance
            prev_performance = current_performance

        shapley_values += marginal_contributions
        print(
            f"Permutation: {perm}, Marginal Contributions: {marginal_contributions}, TMC Shapley Values: {shapley_values}"
        )

    return shapley_values / num_permutations


# Compute true Shapley values
def compute_true_shapley(clients):
    shapley_values = np.zeros(NUM_CLIENTS)
    all_permutations = list(permutations(range(NUM_CLIENTS)))

    for perm in all_permutations:
        marginal_contributions = np.zeros(NUM_CLIENTS)
        prev_performance = 0
        prev_set = set()
        for client in perm:
            model = clients[client]
            current_performance = evaluate_model(model, test_loader)
            marginal_contributions[client] = current_performance - prev_performance
            prev_set.add(client)
            prev_performance = current_performance

        shapley_values += marginal_contributions
        print(
            f"Permutation: {perm}, Marginal Contributions: {marginal_contributions}, True Shapley Values: {shapley_values}"
        )

    return shapley_values / len(all_permutations)


coalitions = make_coalitions(NUM_CLIENTS)
print(coalitions)

# Define data splits

num_classes = 10

# client_data_indices = generate_disjoint_labels_datasets(NUM_CLIENTS, num_classes, dataset, NUM_CLS_PER_CLIENT)

client_data_indices = generate_client_data_non_iid(NUM_CLIENTS, num_classes, dataset)


test_vals_coalitions, val_coalitions = train_coalitions(coalitions, client_data_indices)

true_shapley_values = compute_true_shapley_og(test_vals_coalitions)


# clients, train_loaders, val_loaders, test_evals = train_single_clients()


# exit()

# shapley_values = compute_tmc_shapley(clients)
# true_shapley_values = compute_true_shapley(clients)
# shapley_values = compute_tmc_shapley()
# print(shapley_values)
# print(true_shapley_values)

path = Path("results") / EXP_NAME
path.mkdir(parents=True, exist_ok=True)

exp_cfg = {
    "num_clients": NUM_CLIENTS,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    # "peer_evals": PEER_EVALS,
    "NUM_CLS_PER_CLIENT": NUM_CLS_PER_CLIENT,
    # "data_skew": DATA_SKEW,
    "seed": SEED,
}
with open(path / "exp_cfg.yaml", "w") as file:
    yaml.dump(exp_cfg, file)
np.save(path / "approx_shapley_values_niid.npy", shapley_values)
np.save(path / "true_shapley_values_niid.npy", true_shapley_values)
np.save(path / "test_evals_niid.npy", test_evals)
# Compare Reputation Scores vs. Shapley Values
import matplotlib.pyplot as plt

# client_ids = list(reputation_scores.keys())
plt.figure()
plt.scatter(true_shapley_values, shapley_values, label="Clients")
plt.xlabel("True Shapley")
plt.ylabel("TMC-Shapley Value")
plt.title("Comparison of TMC vs. True Shapley Values")
plt.legend()
plt.savefig(path / "approx_vs_trueshapley_niid.png")

plt.figure()
plt.scatter(test_evals, shapley_values, label="Clients")
plt.xlabel("Test accuracies")
plt.ylabel("TMC-Shapley Value")
plt.title("Comparison of TMC shapley values vs. Test accuracies")
plt.legend()
plt.savefig(path / "tmcshapley_vs_testacc_niid.png")

plt.figure()
plt.scatter(test_evals, true_shapley_values, label="Clients")
plt.xlabel("Test accuracies")
plt.ylabel("True Shapley Value")
plt.title("Comparison of True shapley values vs. Test accuracies")
plt.legend()
plt.savefig(path / "trueshapley_vs_testacc_niid.png")
