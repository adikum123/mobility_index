from src.engine import ANFIS

for mf_type in [
    "gaussian",
    "gaussian2",
    "bell",
    "triangular",
    "trapezoidal",
    "pi",
    "diffsigmoidal",
    "prodsigmoidal",
    "sigmoidal",
    "sshape",
    "zshape",
    "linsshape",
    "linzshape",
]:
    for num_indices in [3, 4]:
        anfis = ANFIS(
            num_indices=num_indices,
            num_epochs=5,
            learning_rate=0.01,
            membership_functions=mf_type,
            loss_function="rmse",
            sheet_dir="real",
            verbose=True
        )
        anfis.train()
        anfis.test()