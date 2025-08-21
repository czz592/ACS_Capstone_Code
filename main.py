import time
import json
from collections import OrderedDict

import energy_tracker
import flwr as fl
from flwr.common import Metrics, Context, Parameters, ndarrays_to_parameters
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


from client import IoTClient
from dataset import preprocessing
from server import FedAvgCustom, get_evaluate_fn
from model.architecture import DNN, CNN
from model.training import train_model, test_model
from model.explainability import explain_with_shap, explain_with_captum

# hardcoding input_dim to initiate net before creating clients
# so that server can access it for global model initialization
# used for eval purposes
train_loader = torch.load(
    "train_loader.pth", map_location="cuda:0", weights_only=False)
test_loader = torch.load(
    "test_loader.pth",  map_location="cuda:0", weights_only=False)
input_dim = train_loader.dataset[0][0].shape[0]
num_rounds = 3
num_clients = 5
iid = True


def client_fn(context: Context):
    # cid = int(context.node_id)
    cid = int(context.node_config["partition-id"])
    data_path = f'dataset/edge-iiotset/partitions_{"iid" if iid else "non-iid"}/{num_clients}_clients/client_{cid+1}.csv'

    df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)
    X = df.drop(columns=["Attack_label"])
    y = df["Attack_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = preprocessing.tensorise_and_wrap(X_train, y_train)
    test_loader = preprocessing.tensorise_and_wrap(X_test, y_test)

    return IoTClient(cid=cid, model="CNN", train_loader=train_loader, test_loader=test_loader, dp=True).to_client()


def server_fn(context: Context):
    model = CNN(input_dim=input_dim)
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    global_model_init = ndarrays_to_parameters(params)
    config = ServerConfig(num_rounds=num_rounds)
    strategy = FedAvgCustom(
        file_name=f"{num_clients}_clients" if iid else f"{num_clients}_clients_non_iid",
        num_rounds=num_rounds,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        initial_parameters=global_model_init,  # initialised global model
        evaluate_fn=get_evaluate_fn(
            testloader=test_loader, input_dim=input_dim),
    )

    return ServerAppComponents(strategy=strategy, config=config)


def main(federate=True):
    if federate:
        assert num_clients in [3, 5, 10]
        """fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=5,
            config=fl.server.ServerConfig(num_rounds=federation_rounds)
        )"""
        server = ServerApp(server_fn=server_fn)
        client = ClientApp(client_fn=client_fn)

        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=num_clients
        )
    else:
        # run centralised pipeline
        # train_loader, test_loader, _ = preprocessing.preprocess()
        global train_loader
        global test_loader
        """
        for learning_rate in [0.001, 0.01, 0.1]:
            model = DNN(input_dim=input_dim)
            train_model(model, train_loader, epochs=5,
                        dp=False, lr=learning_rate)
            loss, metrics = test_model(
                model, test_loader, global_model=True)
            print(f"Loss: {loss} | Metrics: {metrics}")
            # write metrics to output/centralised/{file_name}.csv
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
            metrics_df.to_csv(
                f"output/centralised/lr_{learning_rate}_dnn_metrics.csv")
            model = CNN(input_dim=input_dim)
            train_model(model, train_loader, epochs=5,
                        dp=False, lr=learning_rate)
            loss, metrics = test_model(
                model, test_loader, global_model=True)
            print(f"Loss: {loss} | Metrics: {metrics}")
            # write metrics to output/centralised/{file_name}.csv
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
            metrics_df.to_csv(
                f"output/centralised/lr_{learning_rate}_cnn_metrics.csv")"""

        # train dnn with dp
        for noise_multiplier in [0.25, 0.5, 0.75, 1.0]:
            model = DNN(input_dim=input_dim)
            train_model(model, train_loader, epochs=5,
                        dp=True, noise_multiplier=noise_multiplier)
            loss, metrics = test_model(
                model, test_loader, global_model=True)
            # write metrics to output/centralised/{file_name}.csv
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
            metrics_df.insert(0, "noise_multiplier", noise_multiplier)
            metrics_df.to_csv(
                f"output/centralised/noise_{noise_multiplier}_dnn_metrics.csv")
            model = CNN(input_dim=input_dim)
            train_model(model, train_loader, epochs=5,
                        dp=True, noise_multiplier=noise_multiplier)
            loss, metrics = test_model(
                model, test_loader, global_model=True)
            # write metrics to output/centralised/{file_name}.csv
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
            metrics_df.insert(0, "noise_multiplier", noise_multiplier)
            metrics_df.to_csv(
                f"output/centralised/noise_{noise_multiplier}_cnn_metrics.csv")

        # # explain model
        # explain_with_shap(model, test_loader)
        # explain_with_captum(model, test_loader)


if __name__ == "__main__":
    # experiment setup:
    """
    | Model Type     | DP  | FL  | XAI | Notes                           |
    | -------------- | --- | --- | --- | -----------------------------   |
    | Centralized    | No  | No  | No  | Baseline                        |
    | Centralized    | Yes | No  | No  | DP impact                       |
    | Federated      | No  | Yes | No  | FL overhead vs. central         |
    | Federated      | Yes | Yes | No  | DP + FL performance tradeoff    |
    | Federated      | Yes | Yes | Yes | XAI interpretability snapshot   |
    | (Optional) Any | Any | Any | Yes | XAI interpretability comparison |
    """
    # main(federate=False)
    main(federate=True)
