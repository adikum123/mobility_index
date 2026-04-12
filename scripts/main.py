from src.engine import ANFIS

model = ANFIS(
    num_indices=4,
    num_epochs=10,
    learning_rate=0.5,
    membership_functions="triangular",
    optimizer="hybrid",
    time_interval=3,
    loss_function="mse",
    index4_mode="destination",
    batch_size=256,
    lr_schedule="exponential",
    num_experts=2,
    decay_rate=0.95,
)
model.train()
metrics = model.test()
print(metrics)
