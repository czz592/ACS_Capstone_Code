import flwr as fl
import torch

from model.architecture import DNN, CNN
from model.training import train_model, test_model
import dataset.preprocessing as preprocessing

from collections import OrderedDict


class IoTClient(fl.client.NumPyClient):
    def __init__(self, cid: int, model: str = "CNN"):
        self.cid = cid
        self.train_loader, self.test_loader, _ = preprocessing.preprocess()
        input_dim = self.train_loader.dataset[0][0].shape[0]
        if model == "DNN":
            self.model = DNN(input_dim=input_dim)
        elif model == "CNN":
            self.model = CNN(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown model: {model}")

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_model(self.model, self.train_loader, epochs=1, dp=True)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test_model(self.model, self.test_loader)
        return float(loss), {"accuracy": float(acc)}
