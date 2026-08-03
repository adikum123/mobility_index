"""
Direct diagnostic for the least-squares matrix singularity problem.
This simulates what the hybrid optimizer does during training.
"""
import numpy as np
import warnings

warnings.filterwarnings('error')  # Convert warnings to errors for debugging

def diagnose_least_squares_problem():
    """Test if the LSE matrix becomes singular during training."""
    
    print("\n" + "="*70)
    print("LEAST-SQUARES MATRIX CONDITIONING DIAGNOSTIC")
    print("="*70)
    
    # Simulate what happens in ANFIS hybrid optimizer
    # A is the augmented data matrix (fuzzy outputs + bias term)
    # Shape: (n_samples, n_rules + 1)
    
    print("\n1. TESTING DIFFERENT MATRIX CONDITIONS")
    print("-" * 70)
    
    test_cases = [
        ("Small dataset, 3 rules", n_samples=378, n_rules=3),
        ("Small dataset, 9 rules (3x3)", n_samples=378, n_rules=9),
        ("Medium dataset, 3 rules", n_samples=1134, n_rules=3),
        ("Medium dataset, 27 rules (3x3x3)", n_samples=1134, n_rules=27),
    ]
    
    for case_name, n_samples, n_rules in test_cases:
        print(f"\n   📊 {case_name}")
        print(f"      Samples: {n_samples}, Rules: {n_rules}")
        
        # Create a realistic fuzzy output matrix
        # Fuzzy rules typically produce outputs in [0, 1]
        np.random.seed(42)
        X_aug = np.random.uniform(0, 1, (n_samples, n_rules))
        X_aug = np.hstack([X_aug, np.ones((n_samples, 1))])  # Add bias
        y = np.random.uniform(0.2, 0.8, n_samples)  # Normalized targets
        
        # Compute condition number
        ATA = X_aug.T @ X_aug
        cond_number = np.linalg.cond(ATA)
        
        print(f"      Condition number of A^T A: {cond_number:.2e}")
        if cond_number > 1e10:
            print(f"      ⚠️  SEVERE: Matrix is nearly singular!")
        elif cond_number > 1e8:
            print(f"      ⚠️  POOR: Matrix is ill-conditioned")
        else:
            print(f"      ✓ GOOD: Matrix is well-conditioned")
        
        # Try to solve the system
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                theta = np.linalg.solve(ATA, X_aug.T @ y)
                
                if len(w) > 0:
                    for warning in w:
                        print(f"      ⚠️  {warning.category.__name__}: {warning.message}")
                
                if np.isnan(theta).any() or np.isinf(theta).any():
                    print(f"      ❌ Solution contains NaN/Inf!")
                else:
                    print(f"      ✓ Solution is valid")
        except Exception as e:
            print(f"      ❌ FAILED: {e}")
    
    print("\n\n2. TESTING REGULARIZATION IMPACT")
    print("-" * 70)
    
    # Create a problematic matrix
    np.random.seed(42)
    n_samples, n_rules = 1134, 27
    X_aug = np.random.uniform(0, 1, (n_samples, n_rules + 1))
    X_aug[:, -1] = 1  # Ensure bias column
    y = np.random.uniform(0.2, 0.8, n_samples)
    
    regularization_values = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    for reg in regularization_values:
        ATA_reg = X_aug.T @ X_aug + reg * np.eye(X_aug.shape[1])
        cond_number = np.linalg.cond(ATA_reg)
        
        try:
            theta = np.linalg.solve(ATA_reg, X_aug.T @ y)
            is_valid = not (np.isnan(theta).any() or np.isinf(theta).any())
            status = "✓ Valid" if is_valid else "❌ Invalid"
        except Exception as e:
            cond_number = np.inf
            status = "❌ Failed"
        
        print(f"   Regularization λ={reg:8.0e}: cond={cond_number:.2e} | {status}")


def diagnose_fuzzy_outputs():
    """Check if fuzzy layer outputs could be problematic."""
    
    print("\n\n" + "="*70)
    print("FUZZY LAYER OUTPUT DIAGNOSIS")
    print("="*70)
    
    print("\n1. TYPICAL FUZZY MEMBERSHIP VALUES")
    print("-" * 70)
    
    # Simulate fuzzy fuzzification outputs
    n_samples = 378
    n_mfs = 3  # 3 membership functions per input
    n_inputs = 3  # For 3-indicator model
    n_rules = n_mfs ** n_inputs  # 27 rules
    
    print(f"   3-indicator model: {n_inputs} inputs × {n_mfs} MFs each = {n_rules} rules")
    
    # Simulate fuzzy outputs (T-norm aggregation typically gives small values)
    np.random.seed(42)
    fuzzy_outputs = np.random.uniform(0, 1, (n_samples, n_rules))
    
    # Apply T-norm (product): combines membership from all inputs
    # Result: often very small numbers
    fuzzy_outputs = np.prod(
        np.random.uniform(0.3, 0.9, (n_samples, n_inputs)),
        axis=1,
        keepdims=True
    )
    fuzzy_outputs = np.tile(fuzzy_outputs, (1, n_rules))
    
    print(f"   Fuzzy output matrix shape: {fuzzy_outputs.shape}")
    print(f"   Value range: [{fuzzy_outputs.min():.6f}, {fuzzy_outputs.max():.6f}]")
    print(f"   Mean: {fuzzy_outputs.mean():.6f}, Std: {fuzzy_outputs.std():.6f}")
    
    if fuzzy_outputs.max() < 0.1:
        print(f"   ⚠️  WARNING: Fuzzy outputs are very small!")
        print(f"      This can make A^T A ill-conditioned")


def main():
    diagnose_least_squares_problem()
    diagnose_fuzzy_outputs()
    
    print("\n\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
1. If condition number is > 1e8:
   - Add regularization: λ = 1e-4 to 1e-3
   - Reduce number of membership functions (n_mfs=2)
   - Increase batch size

2. If fuzzy outputs are very small (< 0.1):
   - Scale inputs to [0, 1] properly
   - Check membership function initialization

3. If matrix is fundamentally ill-conditioned:
   - Try SGD optimizer instead of hybrid
   - Use gradient accumulation
   - Reduce learning rate
    """)


if __name__ == "__main__":
    main()
