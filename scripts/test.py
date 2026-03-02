from src.engine import ANFIS

metrics_by_time_interval = {}
for time_interval in range(8):
    print(f"Training and testing model for time interval {time_interval}")
    model = ANFIS(
        num_indices=3,
        num_epochs=10,
        learning_rate=0.01,
        membership_functions="triangular",
        optimizer="hybrid",
        time_interval=time_interval,
        loss_function="mse",
    )
    model.train()
    metrics_by_time_interval[time_interval] = model.test()
    break

print(metrics_by_time_interval)
