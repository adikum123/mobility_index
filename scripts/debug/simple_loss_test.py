"""
Simple test: Compare loss BEFORE and AFTER training
Shows baseline metrics to understand what's happening.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))

from src.engine import ANFIS

def compute_metrics(y_true, y_pred):
    """Calculate MSE and R2."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

print("="*70)
print("LOSS TEST: Baseline vs After Training")
print("="*70)

# Test 1: Current configuration
print("\n1️⃣  CURRENT CONFIG (n_mfs=3, hybrid optimizer)")
print("-"*70)

model1 = ANFIS(
    num_epochs=10,
    learning_rate=0.001,
    membership_functions="triangular",
    loss_function="mse",
    optimizer="hybrid",
    shuffle=True,
    n_mfs=3,
    num_indices=3
)

model1._set_data_expert_split()
print(f"Training on {len(model1.X_train)} samples, validating on {len(model1.X_val)} samples...\n")

# Get initial predictions BEFORE training
print("📊 BASELINE (before training):")
try:
    y_pred_train_init = model1.model.predict(model1.X_train)
    y_pred_val_init = model1.model.predict(model1.X_val)
    
    metrics_train_init = compute_metrics(model1.Y_train, y_pred_train_init)
    metrics_val_init = compute_metrics(model1.Y_val, y_pred_val_init)
    
    print(f"   Train - MSE: {metrics_train_init['mse']:.8f}, R²: {metrics_train_init['r2']:.6f}, RMSE: {metrics_train_init['rmse']:.8f}")
    print(f"   Val   - MSE: {metrics_val_init['mse']:.8f}, R²: {metrics_val_init['r2']:.6f}, RMSE: {metrics_val_init['rmse']:.8f}")
except Exception as e:
    print(f"   Error getting baseline predictions: {e}")
    metrics_train_init = None
    metrics_val_init = None

# Train
print(f"\nTraining...")
model1.model.fit(
    model1.X_train,
    model1.Y_train,
    validation_data=(model1.X_val, model1.Y_val),
    validation_frequency=1,
    verbose=True,
)

# Get metrics AFTER training
print("\n📊 AFTER TRAINING:")
y_pred_train_final = model1.model.predict(model1.X_train)
y_pred_val_final = model1.model.predict(model1.X_val)

metrics_train_final = compute_metrics(model1.Y_train, y_pred_train_final)
metrics_val_final = compute_metrics(model1.Y_val, y_pred_val_final)

print(f"   Train - MSE: {metrics_train_final['mse']:.8f}, R²: {metrics_train_final['r2']:.6f}, RMSE: {metrics_train_final['rmse']:.8f}")
print(f"   Val   - MSE: {metrics_val_final['mse']:.8f}, R²: {metrics_val_final['r2']:.6f}, RMSE: {metrics_val_final['rmse']:.8f}")

# Compare
if metrics_train_init:
    print("\n📈 CHANGE DURING TRAINING:")
    mse_change = metrics_train_final['mse'] - metrics_train_init['mse']
    r2_change = metrics_train_final['r2'] - metrics_train_init['r2']
    print(f"   Train MSE change: {mse_change:+.8f}")
    print(f"   Train R² change:  {r2_change:+.6f}")

history1 = model1.model.training_history_
loss1_start = history1['train'][0]
loss1_end = history1['train'][-1]
improvement1 = loss1_start - loss1_end
pct1 = (improvement1 / loss1_start * 100) if loss1_start != 0 else 0

print(f"\n📊 LOSS HISTORY:")
print(f"   Epoch 1:  {loss1_start:.8f}")
print(f"   Epoch 10: {loss1_end:.8f}")
print(f"   Improvement: {improvement1:.2e} ({pct1:+.4f}%)")

if pct1 < 0.1:
    print(f"   ❌ STAGNATED (loss not improving)")
else:
    print(f"   ✓ IMPROVING")

# Test 2: Try SGD
print("\n\n2️⃣  TRY SGD OPTIMIZER (n_mfs=3)")
print("-"*70)

model2 = ANFIS(
    num_epochs=10,
    learning_rate=0.001,
    membership_functions="triangular",
    loss_function="mse",
    optimizer="sgd",  # Change optimizer
    shuffle=True,
    n_mfs=3,
    num_indices=3
)

model2._set_data_expert_split()
print(f"Training on {len(model2.X_train)} samples, validating on {len(model2.X_val)} samples...\n")

# Get initial predictions BEFORE training
print("📊 BASELINE (before training):")
try:
    y_pred_train_init = model2.model.predict(model2.X_train)
    y_pred_val_init = model2.model.predict(model2.X_val)
    
    metrics_train_init = compute_metrics(model2.Y_train, y_pred_train_init)
    metrics_val_init = compute_metrics(model2.Y_val, y_pred_val_init)
    
    print(f"   Train - MSE: {metrics_train_init['mse']:.8f}, R²: {metrics_train_init['r2']:.6f}, RMSE: {metrics_train_init['rmse']:.8f}")
    print(f"   Val   - MSE: {metrics_val_init['mse']:.8f}, R²: {metrics_val_init['r2']:.6f}, RMSE: {metrics_val_init['rmse']:.8f}")
except Exception as e:
    print(f"   Error getting baseline predictions: {e}")
    metrics_train_init = None
    metrics_val_init = None

# Train
print(f"\nTraining...")
model2.model.fit(
    model2.X_train,
    model2.Y_train,
    validation_data=(model2.X_val, model2.Y_val),
    validation_frequency=1,
    verbose=True,
)

# Get metrics AFTER training
print("\n📊 AFTER TRAINING:")
y_pred_train_final = model2.model.predict(model2.X_train)
y_pred_val_final = model2.model.predict(model2.X_val)

metrics_train_final = compute_metrics(model2.Y_train, y_pred_train_final)
metrics_val_final = compute_metrics(model2.Y_val, y_pred_val_final)

print(f"   Train - MSE: {metrics_train_final['mse']:.8f}, R²: {metrics_train_final['r2']:.6f}, RMSE: {metrics_train_final['rmse']:.8f}")
print(f"   Val   - MSE: {metrics_val_final['mse']:.8f}, R²: {metrics_val_final['r2']:.6f}, RMSE: {metrics_val_final['rmse']:.8f}")

# Compare
if metrics_train_init:
    print("\n📈 CHANGE DURING TRAINING:")
    mse_change = metrics_train_final['mse'] - metrics_train_init['mse']
    r2_change = metrics_train_final['r2'] - metrics_train_init['r2']
    print(f"   Train MSE change: {mse_change:+.8f}")
    print(f"   Train R² change:  {r2_change:+.6f}")

history2 = model2.model.training_history_
loss2_start = history2['train'][0]
loss2_end = history2['train'][-1]
improvement2 = loss2_start - loss2_end
pct2 = (improvement2 / loss2_start * 100) if loss2_start != 0 else 0

print(f"\n📊 LOSS HISTORY:")
print(f"   Epoch 1:  {loss2_start:.8f}")
print(f"   Epoch 10: {loss2_end:.8f}")
print(f"   Improvement: {improvement2:.2e} ({pct2:+.4f}%)")

if pct2 < 0.1:
    print(f"   ❌ STAGNATED or DIVERGED")
else:
    print(f"   ✓ IMPROVING")

# Test 3: Lower learning rate with hybrid
print("\n\n3️⃣  LOWER LEARNING RATE (LR=0.0001, hybrid, n_mfs=3)")
print("-"*70)

model3 = ANFIS(
    num_epochs=10,
    learning_rate=0.0001,  # Much lower
    membership_functions="triangular",
    loss_function="mse",
    optimizer="hybrid",
    shuffle=True,
    n_mfs=3,
    num_indices=3
)

model3._set_data_expert_split()
print(f"Training on {len(model3.X_train)} samples, validating on {len(model3.X_val)} samples...\n")

# Get initial predictions BEFORE training
print("📊 BASELINE (before training):")
try:
    y_pred_train_init = model3.model.predict(model3.X_train)
    y_pred_val_init = model3.model.predict(model3.X_val)
    
    metrics_train_init = compute_metrics(model3.Y_train, y_pred_train_init)
    metrics_val_init = compute_metrics(model3.Y_val, y_pred_val_init)
    
    print(f"   Train - MSE: {metrics_train_init['mse']:.8f}, R²: {metrics_train_init['r2']:.6f}, RMSE: {metrics_train_init['rmse']:.8f}")
    print(f"   Val   - MSE: {metrics_val_init['mse']:.8f}, R²: {metrics_val_init['r2']:.6f}, RMSE: {metrics_val_init['rmse']:.8f}")
except Exception as e:
    print(f"   Error getting baseline predictions: {e}")
    metrics_train_init = None
    metrics_val_init = None

# Train
print(f"\nTraining...")
model3.model.fit(
    model3.X_train,
    model3.Y_train,
    validation_data=(model3.X_val, model3.Y_val),
    validation_frequency=1,
    verbose=True,
)

# Get metrics AFTER training
print("\n📊 AFTER TRAINING:")
y_pred_train_final = model3.model.predict(model3.X_train)
y_pred_val_final = model3.model.predict(model3.X_val)

metrics_train_final = compute_metrics(model3.Y_train, y_pred_train_final)
metrics_val_final = compute_metrics(model3.Y_val, y_pred_val_final)

print(f"   Train - MSE: {metrics_train_final['mse']:.8f}, R²: {metrics_train_final['r2']:.6f}, RMSE: {metrics_train_final['rmse']:.8f}")
print(f"   Val   - MSE: {metrics_val_final['mse']:.8f}, R²: {metrics_val_final['r2']:.6f}, RMSE: {metrics_val_final['rmse']:.8f}")

# Compare
if metrics_train_init:
    print("\n📈 CHANGE DURING TRAINING:")
    mse_change = metrics_train_final['mse'] - metrics_train_init['mse']
    r2_change = metrics_train_final['r2'] - metrics_train_init['r2']
    print(f"   Train MSE change: {mse_change:+.8f}")
    print(f"   Train R² change:  {r2_change:+.6f}")

history3 = model3.model.training_history_
loss3_start = history3['train'][0]
loss3_end = history3['train'][-1]
improvement3 = loss3_start - loss3_end
pct3 = (improvement3 / loss3_start * 100) if loss3_start != 0 else 0

print(f"\n📊 LOSS HISTORY:")
print(f"   Epoch 1:  {loss3_start:.8f}")
print(f"   Epoch 10: {loss3_end:.8f}")
print(f"   Improvement: {improvement3:.2e} ({pct3:+.4f}%)")

if pct3 < 0.1:
    print(f"   ❌ STAGNATED")
else:
    print(f"   ✓ IMPROVING")

# Summary
print("\n\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nConfig 1 (Hybrid LR=0.001, n_mfs=3): {pct1:+7.4f}% improvement")
print(f"Config 2 (SGD LR=0.001, n_mfs=3):    {pct2:+7.4f}% improvement")
print(f"Config 3 (Hybrid LR=0.0001, n_mfs=3):{pct3:+7.4f}% improvement")

improvements = [
    ("Hybrid LR=0.001 n_mfs=3", pct1),
    ("SGD LR=0.001 n_mfs=3", pct2),
    ("Hybrid LR=0.0001 n_mfs=3", pct3),
]
improvements.sort(key=lambda x: -x[1])

print(f"\n🏆 BEST: {improvements[0][0]} ({improvements[0][1]:+.4f}%)")

if improvements[0][1] > 0.1:
    print(f"\n✅ This config shows improvement!")
else:
    print(f"\n⚠️  All configs stagnate. The issue is deeper than just hyperparameters.")