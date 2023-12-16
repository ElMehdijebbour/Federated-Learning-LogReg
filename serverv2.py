import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load heart disease test data
    _, (X_test, y_test) = utils.load_heart_disease_data()

    # Ensure X_test is in the correct format (e.g., NumPy array)
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)

        # Debugging: Check shapes
        assert X_test.shape[1] == model.coef_.shape[1], (
            f"Mismatch in the number of features. "
            f"Expected {model.coef_.shape[1]}, got {X_test.shape[1]}"
        )

        # Perform evaluation
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for federated learning with the heart disease dataset
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
