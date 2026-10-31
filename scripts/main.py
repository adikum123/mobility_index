from src.engine import ANFIS

for num_indices in [3, 4]:
    anfis = ANFIS(
        num_indices=num_indices,
        num_epochs=5,
        learning_rate=0.01,
        membership_functions="gaussian",
        loss_function="rmse",
        sheet_dir="real",
        verbose=True
    )
    anfis.train()
    anfis.test()