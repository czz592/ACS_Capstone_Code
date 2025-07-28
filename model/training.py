# DL
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# PETs
from opacus import PrivacyEngine


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    dp: bool = False,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    grad_sample_mode: str = "ghost"
) -> nn.Module:
    """
    Train a PyTorch model with optional Differential Privacy.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        dp (bool): Whether to enable differential privacy.
        noise_multiplier (float): DP noise multiplier (if dp=True).
        max_grad_norm (float): DP clipping norm (if dp=True).
        grad_sample_mode (str): 'ghost' or 'hooks' (Opacus modes).

    Returns:
        nn.Module: Trained model.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, criterion, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            grad_sample_mode=grad_sample_mode
        )

    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch} | Loss: {loss.item():.4f}")

    return model


def test_model(
    model: nn.Module,
    test_loader: DataLoader
) -> float:
    """
    Evaluate a model on the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data.

    Returns:
        float: Accuracy
    """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            loss_total += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    accuracy = correct / total * 100
    avg_loss = loss_total / len(test_loader)
    print(f"Test Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")
    return accuracy
