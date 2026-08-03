"""
Debug script to inspect why ANFIS training stagnates.
Run this to get visibility into model state during training.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Adjust imports based on your project structure
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.engine import ANFIS

def inspect_model_state(model, X_train, Y_train, X_val, Y_val):
    """Inspect ANFIS model internals after first epoch."""
    print("\n" + "="*70)
    print("ANFIS MODEL INTERNAL STATE INSPECTION")
    print("="*70)
    
    # Inspect model structure
    print("\n1. MODEL STRUCTURE & ATTRIBUTES")
    print("-" * 70)
    anfis_model = model.model
    print(f"   Model type: {type(anfis_model).__name__}")
    print(f"   Available attributes: {[attr for attr in dir(anfis_model) if not attr.startswith('_')][:10]}...")
    
    # Try to access parameters/layers
    print("\n2. PARAMETERS & LAYERS")
    print("-" * 70)
    try:
        if hasattr(anfis_model, 'layers'):
            print(f"   Model has {len(anfis_model.layers)} layers")
            for i, layer in enumerate(anfis_model.layers):
                print(f"      Layer {i}: {type(layer).__name__}")
    except Exception as e:
        print(f"   Could not access layers: {e}")
    
    try:
        if hasattr(anfis_model, 'parameters'):
            params = list(anfis_model.parameters())
            print(f"   Total parameters: {len(params)}")
            for i, param in enumerate(params[:5]):  # Show first 5
                if hasattr(param, 'shape'):
                    print(f"      Param {i}: shape={param.shape}, dtype={param.dtype}")
                    param_val = param.detach().numpy() if hasattr(param, 'detach') else param
                    print(f"              min={param_val.min():.6f}, max={param_val.max():.6f}")
                    print(f"              NaN={np.isnan(param_val).any()}, Inf={np.isinf(param_val).any()}")
    except Exception as e:
        print(f"   Could not access parameters: {e}")
    
    # Test predictions
    print("\n3. MODEL PREDICTIONS")
    print("-" * 70)
    try:
        y_pred_train = anfis_model.predict(X_train)
        y_pred_val = anfis_model.predict(X_val)
        
        print(f"   Train predictions:")
        print(f"      Range: [{y_pred_train.min():.6f}, {y_pred_train.max():.6f}]")
        print(f"      Mean: {y_pred_train.mean():.6f}, Std: {y_pred_train.std():.6f}")
        print(f"      Contains NaN: {np.isnan(y_pred_train).any()}")
        print(f"      Contains Inf: {np.isinf(y_pred_train).any()}")
        
        print(f"\n   Val predictions:")
        print(f"      Range: [{y_pred_val.min():.6f}, {y_pred_val.max():.6f}]")
        print(f"      Mean: {y_pred_val.mean():.6f}, Std: {y_pred_val.std():.6f}")
        print(f"      Contains NaN: {np.isnan(y_pred_val).any()}")
        print(f"      Contains Inf: {np.isinf(y_pred_val).any()}")
        
        # Calculate MSE to verify
        from sklearn.metrics import mean_squared_error
        mse_train = mean_squared_error(Y_train, y_pred_train)
        mse_val = mean_squared_error(Y_val, y_pred_val)
        print(f"\n   Calculated MSE:")
        print(f"      Train: {mse_train:.6f}")
        print(f"      Val:   {mse_val:.6f}")
    except Exception as e:
        print(f"   Error making predictions: {e}")
    
    # Training history
    print("\n4. TRAINING HISTORY")
    print("-" * 70)
    if hasattr(anfis_model, 'training_history_'):
        history = anfis_model.training_history_
        print(f"   Epochs completed: {len(history['train'])}")
        print(f"   Train loss history: {history['train']}")
        print(f"   Val loss history:   {history['val']}")
        
        if len(history['train']) > 1:
            loss_change = abs(history['train'][0] - history['train'][-1])
            loss_pct_change = (loss_change / abs(history['train'][0])) * 100 if history['train'][0] != 0 else 0
            print(f"\n   Loss change from epoch 0 to {len(history['train'])-1}:")
            print(f"      Absolute: {loss_change:.2e}")
            print(f"      Percentage: {loss_pct_change:.4f}%")
            
            if loss_change < 1e-7:
                print(f"      ❌ STAGNATED (no improvement)")
            else:
                print(f"      ✓ IMPROVING")
    else:
        print(f"   training_history_ not available yet")
    
    # Check input/output characteristics
    print("\n5. DATA CHARACTERISTICS")
    print("-" * 70)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"   Y_train shape: {Y_train.shape}")
    print(f"   Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")


def main():
    """Test with 3-indicator model."""
    print("\n🔍 DEBUGGING 3-INDICATOR ANFIS MODEL")
    print("="*70)
    
    ANFIS_ARGS = {
        "num_epochs": 3,  # Train for 3 epochs to see pattern
        "learning_rate": 0.001,
        "membership_functions": "triangular",
        "loss_function": "mse",
        "batch_size": 256,
        "optimizer": "hybrid",
        "shuffle": True,
        "n_mfs": 3,
    }
    
    model = ANFIS(**ANFIS_ARGS, num_indices=3)
    
    # Prepare data (same as in ANFIS.train())
    model._set_data_expert_split()
    
    print(f"\nData prepared:")
    print(f"  Train: {model.X_train.shape[0]} samples, {model.X_train.shape[1]} features")
    print(f"  Val: {model.X_val.shape[0]} samples")
    print(f"  Target stats: min={model.Y_train.min():.4f}, max={model.Y_train.max():.4f}")
    
    # Inspect before training
    print("\n📍 STATE BEFORE TRAINING:")
    inspect_model_state(model, model.X_train, model.Y_train, model.X_val, model.Y_val)
    
    # Train for 3 epochs to see pattern
    print("\n" + "="*70)
    print("TRAINING (3 epochs)...")
    print("="*70)
    model.model.fit(
        model.X_train,
        model.Y_train,
        validation_data=(model.X_val, model.Y_val),
        validation_frequency=1,
        verbose=True,
    )
    
    # Inspect after training
    print("\n📍 STATE AFTER TRAINING:")
    inspect_model_state(model, model.X_train, model.Y_train, model.X_val, model.Y_val)


if __name__ == "__main__":
    main()