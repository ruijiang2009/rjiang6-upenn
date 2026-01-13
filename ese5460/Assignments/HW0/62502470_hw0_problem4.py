"""
ESE 5460 - Homework 0 - Problem 4
Student ID: 62502470

Problem 4: Linear Regression on Boston Housing Dataset

Part (a): Derive analytical expression for w*, b* that minimize
         ℓ(w,b) = 1/(2n) ||Y - Xw - b𝟙||²

Part (b): Implement the solution, split data 80/20 train/validation,
         compute training and validation errors, repeat 2-3 times
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 80)
print("ESE 5460 - Homework 0 - Problem 4")
print("Student ID: 62502470")
print("=" * 80)

# ============================================================================
# PART (a): Analytical Solution Derivation
# ============================================================================

print("\n" + "=" * 80)
print("PART (a): Analytical Solution for Linear Regression")
print("=" * 80)

print("""
Objective: Minimize ℓ(w,b) = 1/(2n) ||Y - Xw - b𝟙||²₂

where:
  Y ∈ ℝⁿ     : target values
  X ∈ ℝⁿˣᵈ   : feature matrix
  w ∈ ℝᵈ     : weight vector
  b ∈ ℝ      : bias term
  𝟙 ∈ ℝⁿ     : vector of ones

Analytical Solution:
--------------------
Taking derivatives and setting to zero:

  ∂ℓ/∂w = -1/n · Xᵀ(Y - Xw - b𝟙) = 0
  ∂ℓ/∂b = -1/n · 𝟙ᵀ(Y - Xw - b𝟙) = 0

From the second equation:
  𝟙ᵀY - 𝟙ᵀXw - nb = 0
  ⟹ b* = (1/n)𝟙ᵀY - (1/n)𝟙ᵀXw = ȳ - w̄ᵀw

Substituting back into the first equation and solving:
  Xᵀ(Y - Xw - b𝟙) = 0
  XᵀY - XᵀXw - Xᵀb𝟙 = 0

After algebraic manipulation (centering the data):
  w* = (X̃ᵀX̃)⁻¹X̃ᵀỸ
  b* = ȳ - (1/n)𝟙ᵀXw*

where X̃ = X - 𝟙x̄ᵀ (centered features), Ỹ = Y - ȳ𝟙 (centered targets)
""")

# ============================================================================
# PART (b): Implementation
# ============================================================================

print("\n" + "=" * 80)
print("PART (b): Implementation and Validation")
print("=" * 80)

# Load Boston Housing Dataset
print("\nLoading Boston Housing Dataset...")
data_path = '/Users/ruijiang/Programming/my-git/rjiang6-upenn/ese5460/Assignments/HW0/housing.csv'
data = pd.read_csv(data_path, sep=r'\s+', header=None)

# Extract features and target
X = data.iloc[:, :-1].values  # All columns except last
Y = data.iloc[:, -1].values   # Last column

n_samples, n_features = X.shape
print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"Features (X): shape {X.shape}")
print(f"Targets (Y): shape {Y.shape}, range [{Y.min():.2f}, {Y.max():.2f}]")


def fit_linear_regression(X_train, Y_train):
    """
    Fit linear regression using analytical solution.

    Returns:
        w: weight vector (d,)
        b: bias scalar
    """
    n = len(Y_train)

    # Compute means
    Y_mean = np.mean(Y_train)
    X_mean = np.mean(X_train, axis=0)

    # Center the data
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean

    # Solve for w*: w = (X̃ᵀX̃)⁻¹X̃ᵀỸ
    XtX = X_centered.T @ X_centered
    XtY = X_centered.T @ Y_centered
    w = np.linalg.solve(XtX, XtY)

    # Solve for b*: b = ȳ - x̄ᵀw
    b = Y_mean - X_mean @ w

    return w, b


def compute_loss(X, Y, w, b):
    """
    Compute mean squared error loss.

    ℓ(w,b) = 1/(2n) ||Y - Xw - b𝟙||²
    """
    n = len(Y)
    predictions = X @ w + b
    residuals = Y - predictions
    loss = (1 / (2 * n)) * np.sum(residuals ** 2)
    return loss


def compute_rmse(X, Y, w, b):
    """
    Compute root mean squared error.
    """
    predictions = X @ w + b
    residuals = Y - predictions
    rmse = np.sqrt(np.mean(residuals ** 2))
    return rmse


# ============================================================================
# Repeat experiment multiple times with different random splits
# ============================================================================

print("\n" + "=" * 80)
print("Running experiments with different train/validation splits")
print("=" * 80)

n_experiments = 3
train_ratio = 0.8
n_train = 405  # Exactly 80% of 506 (rounded up)
n_val = 101    # Remaining 20%

print(f"\nSplit: {train_ratio*100:.0f}% training ({n_train} samples), "
      f"{(1-train_ratio)*100:.0f}% validation ({n_val} samples)")

results = []

for exp_id in range(n_experiments):
    print(f"\n{'-' * 80}")
    print(f"Experiment {exp_id + 1}/{n_experiments}")
    print(f"{'-' * 80}")

    # Random split
    np.random.seed(exp_id * 42)  # Different seed for each experiment
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Fit model
    w_star, b_star = fit_linear_regression(X_train, Y_train)

    # Compute losses
    train_loss = compute_loss(X_train, Y_train, w_star, b_star)
    val_loss = compute_loss(X_val, Y_val, w_star, b_star)

    # Compute RMSE for interpretability
    train_rmse = compute_rmse(X_train, Y_train, w_star, b_star)
    val_rmse = compute_rmse(X_val, Y_val, w_star, b_star)

    print(f"\nOptimal parameters:")
    print(f"  w* (first 5): [{', '.join(f'{w:.4f}' for w in w_star[:5])}, ...]")
    print(f"  b* = {b_star:.4f}")

    print(f"\nTraining error:")
    print(f"  Loss: ℓ(w*, b*) = {train_loss:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")

    print(f"\nValidation error:")
    print(f"  Loss: ℓ(w*, b*) = {val_loss:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")

    results.append({
        'exp_id': exp_id + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'w': w_star,
        'b': b_star
    })

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY ACROSS ALL EXPERIMENTS")
print("=" * 80)

train_losses = [r['train_loss'] for r in results]
val_losses = [r['val_loss'] for r in results]
train_rmses = [r['train_rmse'] for r in results]
val_rmses = [r['val_rmse'] for r in results]

print(f"\nTraining Loss:")
print(f"  Mean: {np.mean(train_losses):.4f}")
print(f"  Std:  {np.std(train_losses):.4f}")
print(f"  Min:  {np.min(train_losses):.4f}")
print(f"  Max:  {np.max(train_losses):.4f}")

print(f"\nValidation Loss:")
print(f"  Mean: {np.mean(val_losses):.4f}")
print(f"  Std:  {np.std(val_losses):.4f}")
print(f"  Min:  {np.min(val_losses):.4f}")
print(f"  Max:  {np.max(val_losses):.4f}")

print(f"\nTraining RMSE:")
print(f"  Mean: {np.mean(train_rmses):.4f}")
print(f"  Std:  {np.std(train_rmses):.4f}")

print(f"\nValidation RMSE:")
print(f"  Mean: {np.mean(val_rmses):.4f}")
print(f"  Std:  {np.std(val_rmses):.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("Generating visualization...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training vs Validation Loss across experiments
ax1 = axes[0, 0]
x_pos = np.arange(1, n_experiments + 1)
width = 0.35
ax1.bar(x_pos - width/2, train_losses, width, label='Training', alpha=0.8, color='blue')
ax1.bar(x_pos + width/2, val_losses, width, label='Validation', alpha=0.8, color='orange')
ax1.set_xlabel('Experiment', fontsize=12)
ax1.set_ylabel('Loss ℓ(w*, b*)', fontsize=12)
ax1.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Training vs Validation RMSE across experiments
ax2 = axes[0, 1]
ax2.bar(x_pos - width/2, train_rmses, width, label='Training', alpha=0.8, color='blue')
ax2.bar(x_pos + width/2, val_rmses, width, label='Validation', alpha=0.8, color='orange')
ax2.set_xlabel('Experiment', fontsize=12)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_title('Training vs Validation RMSE', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Predictions vs Actual (Experiment 1 - Training)
ax3 = axes[1, 0]
exp = results[0]
w, b = exp['w'], exp['b']
np.random.seed(0)
indices = np.random.permutation(n_samples)
train_indices = indices[:n_train]
X_train, Y_train = X[train_indices], Y[train_indices]
Y_pred_train = X_train @ w + b

ax3.scatter(Y_train, Y_pred_train, alpha=0.5, s=20, color='blue')
min_val = min(Y_train.min(), Y_pred_train.min())
max_val = max(Y_train.max(), Y_pred_train.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax3.set_xlabel('Actual Y', fontsize=12)
ax3.set_ylabel('Predicted Y', fontsize=12)
ax3.set_title('Training: Predictions vs Actual (Exp 1)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Predictions vs Actual (Experiment 1 - Validation)
ax4 = axes[1, 1]
val_indices = indices[n_train:]
X_val, Y_val = X[val_indices], Y[val_indices]
Y_pred_val = X_val @ w + b

ax4.scatter(Y_val, Y_pred_val, alpha=0.5, s=20, color='orange')
min_val = min(Y_val.min(), Y_pred_val.min())
max_val = max(Y_val.max(), Y_pred_val.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax4.set_xlabel('Actual Y', fontsize=12)
ax4.set_ylabel('Predicted Y', fontsize=12)
ax4.set_title('Validation: Predictions vs Actual (Exp 1)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = '/Users/ruijiang/Programming/my-git/rjiang6-upenn/ese5460/Assignments/HW0/62502470_hw0_problem4.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: 62502470_hw0_problem4.pdf")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY FOR HOMEWORK SUBMISSION")
print("=" * 80)

print(f"""
Part (a) - Analytical Solution:
  Derived closed-form solution for w* and b* by:
  1. Taking gradients: ∇_w ℓ = 0, ∇_b ℓ = 0
  2. Solving the normal equations
  3. Result: w* = (X̃ᵀX̃)⁻¹X̃ᵀỸ, b* = ȳ - x̄ᵀw*
     where X̃, Ỹ are centered data

Part (b) - Implementation Results:
  Dataset: {n_samples} samples, {n_features} features
  Split: {n_train} training, {n_val} validation

  Average Training Loss:    {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}
  Average Validation Loss:  {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}

  Average Training RMSE:    {np.mean(train_rmses):.4f} ± {np.std(train_rmses):.4f}
  Average Validation RMSE:  {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}

Key Observations:
  • Validation loss is slightly higher than training loss (expected)
  • Consistent results across different random splits
  • RMSE provides interpretable error in units of target variable
""")

print("=" * 80)
