from collections import OrderedDict

import pandas as pd
from flwr.common import Metrics, Parameters, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import matplotlib.pyplot as plt
import torch

from model.architecture import DNN, CNN
from model.training import test_model
from model.explainability import explain_with_shap, explain_with_captum


class FedAvgCustom(FedAvg):
    def __init__(self, file_name: str, num_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name
        self.num_rounds = num_rounds
        self.loss_list = []
        self.metrics_list = []

    def _make_plot(self):
        if not self.loss_list or not self.metrics_list:
            print("No data to plot.")
            return

        try:
            # Plot accuracy over rounds
            rounds = list(range(1, len(self.metrics_list) + 1))
            acc = [metrics.get("accuracy", 0) for metrics in self.metrics_list]
            plt.figure()
            plt.plot(rounds, acc, label='Accuracy')
            plt.grid()
            plt.ylabel("Accuracy (%)")
            plt.xlabel("Round")
            plt.title('Accuracy over rounds')
            plt.legend()
            plt.savefig("output/federated/" + self.file_name + '_accuracy.png')
            plt.close()

            # Plot metrics over rounds
            plt.figure()
            for metric_name, metric_values in self.metrics_list[0].items():
                plt.plot(rounds, [metrics.get(metric_name, 0) for metrics in self.metrics_list], label=metric_name)
            plt.xlabel('Rounds')
            plt.ylabel('Metrics')
            plt.title('Metrics over rounds')
            plt.legend()
            # save to output/federated/{file_name}_metrics.png
            plt.savefig("output/federated/" + self.file_name + '_metrics.png')
            plt.close()

        except Exception as e:
            print(f"An error occurred while plotting: {e}")

    def evaluate(self, server_round: int, parameters: Parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        self.loss_list.append(loss)
        self.metrics_list.append(metrics)
        if server_round == self.num_rounds:
            # Save to CSV
            metrics_df = pd.DataFrame(self.metrics_list)
            metrics_df.to_csv("output/federated/" + self.file_name + ".csv")

            # Generate plot
            self._make_plot()
        return loss, metrics


def get_evaluate_fn(testloader, input_dim):
    def evaluate_fn(server_round: int, parameters, config):
        model = CNN(input_dim=input_dim)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # set params to model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # evaluate model
        print("Beginning global model evaluation...")
        model.eval()
        loss, metrics = test_model(model, testloader, global_model=True)

        return loss, Metrics(metrics)

    return evaluate_fn
