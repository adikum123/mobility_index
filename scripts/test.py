from src.engine import ANFIS

try:
    m = ANFIS(
        num_indices=3, num_epochs=1, learning_rate=0.1,
        membership_functions="definitely_not_a_real_mf_xyz123",
        loss_function="mse", optimizer="hybrid",
        shuffle=True, n_mfs=3, sheet_dir="real",
    )
    m.train()
    print("⚠️  No validation — invalid mf_type was silently accepted!")
except Exception as e:
    print(f"✅ Validated: {type(e).__name__}: {e}")