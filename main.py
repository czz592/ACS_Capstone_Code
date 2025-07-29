import flwr as fl
from client import IoTClient
from dataset import preprocessing
from model.architecture import DNN, CNN
from model.training import train_model, test_model
from model.explainability import explain_with_shap, explain_with_captum


def client_fn(cid):
    return IoTClient(cid=int(cid)).to_client()


def main(federate=True, federation_rounds: int = 5):
    if federate:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=5,
            config=fl.server.ServerConfig(num_rounds=federation_rounds)
        )
    else:
        # run centralised pipeline
        train_loader, test_loader, _ = preprocessing.preprocess()
        model = DNN(input_dim=train_loader.dataset[0][0].shape[0])
        train_model(model, train_loader, epochs=5, dp=True)
        loss, acc = test_model(model, test_loader)
        print(f"Loss: {loss} | Accuracy: {acc}")


if __name__ == "__main__":
    main(federate=True)
