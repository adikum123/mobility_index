from src.engine import ANFIS, MatricesProcessor
from src.index4 import compute_and_save_index4

mp = MatricesProcessor()
mp.compute_distance_matrix()
mp.compute_time_matrices()
mp.compute_journey_count_matrices()

compute_and_save_index4()

model = ANFIS(
    num_indices=4,
    num_epochs=10,
    learning_rate=0.25,
    membership_functions="triangular",
    optimizer="hybrid",
    time_interval=3,
    loss_function="mse",
    index4_mode="average",
    batch_size=32,
)
model.train()
metrics = model.test()
print(metrics)
