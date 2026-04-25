from src.engine import ANFIS

model = ANFIS(
    num_indices=3,
    num_epochs=5,
    learning_rate=0.001,
    membership_functions="triangular",
    time_interval=3,
    loss_function="mse",
    batch_size=256,
    optimizer="adam",
    shuffle=True,
    n_mfs=3,
    num_train_experts=3,
    num_val_experts=2,
    num_test_experts=2,
)
model.train()
metrics = model.test()
print(metrics)
