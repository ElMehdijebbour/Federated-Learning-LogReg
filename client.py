import warnings
import flwr as fl
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils  # Ensure utils contains the modified functions for heart disease dataset

if __name__ == "__main__":
    # Parse command line arguments for user ID
    parser = argparse.ArgumentParser(description='Start a Flower client with specific dataset based on user ID.')
    parser.add_argument('--user_id', type=int, required=True, help='User ID to load specific dataset')
    args = parser.parse_args()
    user_id = args.user_id

    # Load Heart Disease dataset based on the user ID
    (X_train, y_train), (X_test, y_test) = utils.load_heart_disease_data(user_id)

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=10,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
        # Additional hyperparameters might be adjusted based on the dataset
    )

    # Setting initial parameters
    utils.set_initial_params(model)

    # Define Flower client for Heart Disease dataset
    class HeartDiseaseClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=HeartDiseaseClient())
