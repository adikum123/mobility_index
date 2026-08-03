"""
Test different hyperparameter combinations to fix training stagnation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.engine import ANFIS

def test_configuration(config_name, **kwargs):
    """Test a single configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(kwargs, indent=2)}")
    
    model = ANFIS(**kwargs, num_indices=3)
    model._set_data_expert_split()
    
    print(f"\nTraining...")
    model.model.fit(
        model.X_train,
        model.Y_train,
        validation_data=(model.X_val, model.Y_val),
        validation_frequency=1,
        verbose=True,
    )
    
    history = model.model.training_history_
    train_loss = list(history['train'])
    val_loss = list(history['val'])
    
    # Calculate loss improvement
    loss_improvement = abs(train_loss[0] - train_loss[-1])
    improvement_pct = (loss_improvement / train_loss[0]) * 100 if train_loss[0] != 0 else 0
    
    print(f"\n📊 Results:")
    print(f"   Initial loss: {train_loss[0]:.6f}")
    print(f"   Final loss:   {train_loss[-1]:.6f}")
    print(f"   Improvement:  {loss_improvement:.6e} ({improvement_pct:.2f}%)")
    
    if loss_improvement < 1e-7:
        print(f"   ❌ STAGNATED")
    else:
        print(f"   ✓ IMPROVING")
    
    # Test
    metrics = model.test()
    print(f"   R²:   {metrics['r2']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    
    return {
        'name': config_name,
        'loss_improved': loss_improvement > 1e-7,
        'improvement': improvement_pct,
        'r2': metrics['r2'],
        'rmse': metrics['rmse'],
    }


def main():
    """Test various configurations."""
    
    base_config = {
        "num_epochs": 10,
        "membership_functions": "triangular",
        "loss_function": "mse",
        "shuffle": True,
    }
    
    results = []
    
    # Test 1: Reduce learning rate
    print("\n" + "="*70)
    print("TEST 1: LOWER LEARNING RATE")
    print("="*70)
    config = {**base_config, "learning_rate": 0.0001, "batch_size": 256, "optimizer": "hybrid", "n_mfs": 3}
    results.append(test_configuration("Learning Rate 1e-4", **config))
    
    # Test 2: Increase batch size
    print("\n" + "="*70)
    print("TEST 2: LARGER BATCH SIZE")
    print("="*70)
    config = {**base_config, "learning_rate": 0.001, "batch_size": 1024, "optimizer": "hybrid", "n_mfs": 3}
    results.append(test_configuration("Batch Size 1024", **config))
    
    # Test 3: Reduce membership functions
    print("\n" + "="*70)
    print("TEST 3: FEWER MEMBERSHIP FUNCTIONS")
    print("="*70)
    config = {**base_config, "learning_rate": 0.001, "batch_size": 256, "optimizer": "hybrid", "n_mfs": 2}
    results.append(test_configuration("n_mfs=2", **config))
    
    # Test 4: Try SGD optimizer
    print("\n" + "="*70)
    print("TEST 4: SGD OPTIMIZER")
    print("="*70)
    config = {**base_config, "learning_rate": 0.001, "batch_size": 256, "optimizer": "sgd", "n_mfs": 3}
    results.append(test_configuration("SGD Optimizer", **config))
    
    # Test 5: Combination: lower LR + fewer MFs + larger batch
    print("\n" + "="*70)
    print("TEST 5: COMBINATION (Lower LR + n_mfs=2 + Large Batch)")
    print("="*70)
    config = {**base_config, "learning_rate": 0.0001, "batch_size": 1024, "optimizer": "hybrid", "n_mfs": 2}
    results.append(test_configuration("Combined Strategy", **config))
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Configuration':<35} {'Improving?':<12} {'Improvement %':<15} {'R²':<10}")
    print("-" * 70)
    
    for result in results:
        improving = "✓ Yes" if result['loss_improved'] else "❌ No"
        print(f"{result['name']:<35} {improving:<12} {result['improvement']:>6.2f}%    {result['r2']:.6f}")
    
    # Find best
    improving_results = [r for r in results if r['loss_improved']]
    if improving_results:
        best = max(improving_results, key=lambda x: x['improvement'])
        print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
        print(f"   Loss improvement: {best['improvement']:.2f}%")
    else:
        print("\n⚠️  No configuration improved loss!")


if __name__ == "__main__":
    main()
