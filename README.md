# Federated Learning with Logistic Regression on Heart Disease Dataset

This example demonstrates how to perform logistic regression within the Flower framework for federated learning. We are using the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) from the UCI Machine Learning Repository, focusing on a binary classification task to predict the presence of heart disease.

## Project Setup

Start by cloning the example project with the following command:

```shell
git clone https://github.com/ElMehdijebbour/Federated-Learning-LogReg.git
```

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:



This will create a new directory called `Federated-Learning-LogReg` containing the following files:

```
-- README.md         <- Your're reading this right now
-- server.py         <- Defines the server-side logic
-- client.py         <- Defines the client-side logic
-- run.sh            <- Commands to run experiments
-- pyproject.toml    <- Example dependencies (if you use Poetry)
-- requirements.txt  <- Example dependencies
```

### Installing Dependencies

Project dependencies (such as `xgboost` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with XGBoost and Flower

Afterwards you are ready to start the Flower server as well as the clients.
You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning.
To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --node-id=0
```

Start client 2 in the second terminal:

```shell
python3 client.py --node-id=1
```

You will see that Logistic regression is starting a federated training.

Alternatively, you can use `run.sh` to run the same experiment in a single terminal as follows:

```shell
poetry run ./run.sh
```

