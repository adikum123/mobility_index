from src.engine import ANFIS

model = ANFIS(
    num_indices=3,
    num_epochs=5,
    learning_rate=0.01,
    membership_functions="triangular",
    optimizer="adam",
    time_interval=0,
)

model.train()
model.test()
