from src.engine import ANFIS

model = ANFIS(
    num_indices=3, num_epochs=3, learning_rate=0.1,
    membership_functions="triangular", loss_function="mse",
    optimizer="hybrid", shuffle=True, n_mfs=3, sheet_dir="real",
    input_jitter=0.03,   # <-- opt in
)
model.train()
metrics = model.test()
print(metrics)