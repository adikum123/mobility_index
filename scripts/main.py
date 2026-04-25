from src.engine import ANFIS, MatricesProcessor

matrices_processor = MatricesProcessor()
matrices_processor.compute_distance_matrix()
matrices_processor.compute_time_matrices()
matrices_processor.compute_journey_count_matrices()

model = ANFIS(
    num_indices=3,
    num_epochs=5,
    learning_rate=0.01,
    membership_functions="triangular",
    time_interval=3,
    loss_function="mse",
    batch_size=256,
    optimizer="hybrid",
    shuffle=True,
    n_mfs=3,
    num_train_experts=3,
    num_val_experts=2,
    num_test_experts=2,
)
model.train()
metrics = model.test()
print(metrics)
