from src.engine import ANFIS
from src.engine.matrices_processor import MatricesProcessor

matrices_processor = MatricesProcessor()
matrices_processor.compute_distance_matrix()
matrices_processor.compute_time_matrices()
matrices_processor.compute_journey_count_matrices()

model = ANFIS(
    num_indices=3,
    num_epochs=10,
    learning_rate=0.005,
    membership_functions="gaussian",
    time_interval=3,
    loss_function="mse",
    batch_size=256,
    optimizer="adam",
    shuffle=True,
    index4_mode="destination",
    num_experts=2,
    n_mfs=3,
)
model.train()
metrics = model.test()
print(metrics)
