import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load heart disease test data for server-side evaluation
    # Assuming a default or common test dataset for evaluation
    _, (X_test, y_test) = utils.load_heart_disease_data(1)
    print(y_test.shape)
    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        
        # Check if y_test has more than one unique label
        unique_labels = np.unique(y_test)
        if len(unique_labels) < 2:
            raise ValueError(f"y_test contains only one label: {unique_labels}. Evaluation requires at least two different labels.")

        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss,  {"accuracy": accuracy}


    return evaluate

# Start Flower server for federated learning with the heart disease dataset
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )
