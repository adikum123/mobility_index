from src.engine import ANFIS

model = ANFIS(
    num_indices=3,
    num_epochs=5,
    learning_rate=0.1,
    membership_functions="gaussian",
    time_interval=3,
    loss_function="mse",
    index4_mode="destination",
    batch_size=256,
    num_experts=2,
    n_mfs=3,
)
model.train()
metrics = model.test()
print(metrics)
