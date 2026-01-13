import numpy as np

print("=" * 70)
print("Problem 2.a: Verifying Optimal Point")
print("=" * 70)

# Given optimal point from problem 2.a
x_star = (5 - np.sqrt(21)) / 2
y_star = (-3 + np.sqrt(21)) / 2

print(f"\nGiven optimal point from 2.a:")
print(f"x* = (5 - √21)/2 = {x_star:.6f}")
print(f"y* = (-3 + √21)/2 = {y_star:.6f}")

# Objective function: f(x,y) = x^2 + y^2 - 6xy - 4x - 5y
f_star = x_star**2 + y_star**2 - 6*x_star*y_star - 4*x_star - 5*y_star
print(f"\nOptimal loss: f(x*, y*) = {f_star:.6f}")

# Check constraints
# g1: y <= -(x-2)^2 + 4, or equivalently: y + (x-2)^2 - 4 <= 0
# g2: y >= -x + 1, or equivalently: -x + 1 - y <= 0
g1_val = y_star + (x_star - 2)**2 - 4
g2_val = -x_star + 1 - y_star

print(f"\nConstraint verification (both should be ≈ 0 if active):")
print(f"g1(x*, y*) = y + (x-2)² - 4 = {g1_val:.10f}")
print(f"g2(x*, y*) = -x + 1 - y = {g2_val:.10f}")
print("✓ Both constraints are active (binding) at the optimum")

# Compute gradients at the optimal point
print("\n" + "=" * 70)
print("Computing Gradients at Optimal Point")
print("=" * 70)

# ∇f = (∂f/∂x, ∂f/∂y) = (2x - 6y - 4, 2y - 6x - 5)
grad_f = np.array([
    2*x_star - 6*y_star - 4,
    2*y_star - 6*x_star - 5
])

# ∇g1 = (∂g1/∂x, ∂g1/∂y) = (2(x-2), 1)
grad_g1 = np.array([
    2*(x_star - 2),
    1
])

# ∇g2 = (∂g2/∂x, ∂g2/∂y) = (-1, -1)
grad_g2 = np.array([-1, -1])

print(f"\n∇f  = ({grad_f[0]:.6f}, {grad_f[1]:.6f})")
print(f"∇g1 = ({grad_g1[0]:.6f}, {grad_g1[1]:.6f})")
print(f"∇g2 = ({grad_g2[0]:.6f}, {grad_g2[1]:.6f})")

# Find Lagrange multipliers
# At the optimum: ∇f + λ1∇g1 + λ2∇g2 = 0
# This gives us: [∇g1 | ∇g2] * [λ1; λ2] = -∇f
# Solve: A * λ = b, where A = [grad_g1, grad_g2] (as columns), b = -grad_f

print("\n" + "=" * 70)
print("Solving for Lagrange Multipliers")
print("=" * 70)

print("\nKKT stationarity condition: ∇f + λ1∇g1 + λ2∇g2 = 0")
print("Rearranging: [∇g1 | ∇g2] * [λ1; λ2]ᵀ = -∇f")

A = np.column_stack([grad_g1, grad_g2])
b = -grad_f

print(f"\nMatrix equation:")
print(f"A = {A}")
print(f"b = {b}")

lambdas = np.linalg.solve(A, b)
lambda1, lambda2 = lambdas

print(f"\nSolution:")
print(f"λ1 = {lambda1:.6f}")
print(f"λ2 = {lambda2:.6f}")

# Verify the solution
verification = grad_f + lambda1 * grad_g1 + lambda2 * grad_g2
print(f"\nVerification: ∇f + λ1∇g1 + λ2∇g2 = ({verification[0]:.10f}, {verification[1]:.10f})")
print("✓ Should be approximately (0, 0)")

if lambda1 >= 0 and lambda2 >= 0:
    print("\n✓ KKT conditions satisfied: λ1 ≥ 0 and λ2 ≥ 0")
else:
    print(f"\n✗ Warning: KKT condition violated (need λ ≥ 0 for minimization)")

# Problem 2.b: Sensitivity Analysis
print("\n" + "=" * 70)
print("Problem 2.b: Sensitivity Analysis")
print("=" * 70)

print("\nConstraint change:")
print("  Old: y ≤ -(x-2)² + 4")
print("  New: y ≤ -(x-2)² + 4.1")
print(f"  Change: Δc1 = 0.1")

print("\n" + "-" * 70)
print("Interpretation of Lagrange Multiplier λ1:")
print("-" * 70)
print("\nThe Lagrange multiplier λ1 represents the rate of change of")
print("the optimal value with respect to the constraint bound:")
print("\n  df*/dc1 ≈ -λ1")
print("\nPhysically, λ1 tells us how much the minimum loss would improve")
print("if we relaxed the constraint (increased the bound).")

delta_c1 = 0.1
estimated_change = -lambda1 * delta_c1

print("\n" + "-" * 70)
print("Calculation:")
print("-" * 70)
print(f"\nΔf* ≈ -λ1 × Δc1")
print(f"    = -{lambda1:.6f} × {delta_c1}")
print(f"    = {estimated_change:.6f}")

new_optimal_value = f_star + estimated_change

print(f"\nEstimated new optimal value:")
print(f"f*_new ≈ f*_old + Δf*")
print(f"       = {f_star:.6f} + {estimated_change:.6f}")
print(f"       = {new_optimal_value:.6f}")

print("\n" + "-" * 70)
print("Explanation:")
print("-" * 70)
if lambda1 > 0:
    print(f"\nSince λ1 = {lambda1:.6f} > 0, the first constraint is active and")
    print("binding at the optimum. Relaxing it (increasing from 4 to 4.1)")
    print("allows the optimizer to find a better (lower) loss value.")
    print(f"\nThe loss decreases by approximately {-estimated_change:.6f}.")
else:
    print(f"\nSince λ1 = {lambda1:.6f}, the constraint is not binding.")

print("\n" + "=" * 70)
print("Summary for Problem 2.b")
print("=" * 70)
print(f"\n✓ Lagrange multiplier: λ1 = {lambda1:.6f}")
print(f"✓ Estimated change in optimal loss: Δf* ≈ {estimated_change:.6f}")
print(f"✓ New optimal loss: f*_new ≈ {new_optimal_value:.6f}")
print("\nThe gradient of the loss at the stationary point is:")
print(f"  ∇f = {lambda1:.6f} × ∇g1 + {lambda2:.6f} × ∇g2")
print("This shows that ∇f is a linear combination of the constraint")
print("gradients, with the Lagrange multipliers as coefficients.")
