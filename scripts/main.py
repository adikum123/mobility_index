import json
import sys

from src.engine import ANFIS

ANFIS_ARGS = {
    "num_epochs": 10,
    "learning_rate": 0.001,
    "membership_functions": "triangular",
    "loss_function": "mse",
    "optimizer": "hybrid",
    "shuffle": True,
    "n_mfs": 3,
}

for sheet_dir in ["fabricated", "real"]:
    print(f"Using: {sheet_dir} sheets")
    for num_indices in (3, 4):
        model = ANFIS(**ANFIS_ARGS, num_indices=num_indices, sheet_dir=sheet_dir)
        model.train()
        metrics = model.test()
        print(f"num_indices={num_indices}:\n{json.dumps(metrics, indent=4)}")
        sys.exit(1)
