import json

from src.engine import ANFIS

ANFIS_ARGS = {
    "num_epochs": 100,
    "learning_rate": 0.01,
    "membership_functions": "triangular",
    "loss_function": "mse",
    "batch_size": 256,
    "optimizer": "adam",
    "shuffle": True,
    "n_mfs": 3,
}

for num_indices in (3, 4):
    model = ANFIS(**ANFIS_ARGS, num_indices=num_indices)
    model.train()
    metrics = model.test()
    print(f"num_indices={num_indices}:\n{json.dumps(metrics, indent=4)}")
