import flwr as fl
import torch

from model.architecture import DNN, CNN
from model.training import train_model, test_model

from collections import OrderedDict

from codecarbon import EmissionsTracker
from energy_tracker.tracker import EnergyTracker
import time


class IoTClient(fl.client.NumPyClient):
    def __init__(self, cid: int, model: str = "CNN", train_loader=None, test_loader=None, dp: bool = False, noise_multiplier: float = 1.0):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        input_dim = self.train_loader.dataset[0][0].shape[0]
        if model == "DNN":
            self.model = DNN(input_dim=input_dim)
        elif model == "CNN":
            self.model = CNN(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown model: {model}")
        self.dp = dp
        self.noise_multiplier = noise_multiplier

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        tracker = EnergyTracker()
        # track time, energy
        tracker.log_energy_consumption()
        start_time = time.time()
        train_model(self.model, self.train_loader, epochs=1,
                    dp=self.dp, noise_multiplier=self.noise_multiplier)
        end_time = time.time()
        tracker.stop_tracker()
        tracker.shutdown()
        print(f"Training time: {end_time - start_time}")
        # TODO: implement custom tracker to return value
        return self.get_parameters(), len(self.train_loader.dataset), {'time': end_time - start_time, 'energy': 0}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test_model(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}
