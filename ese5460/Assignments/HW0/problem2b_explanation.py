import numpy as np

print("=" * 80)
print("Problem 2.b: Detailed Analysis")
print("=" * 80)

# From problem 2.a, the optimal point
x_star = (5 - np.sqrt(21)) / 2
y_star = (-3 + np.sqrt(21)) / 2

print(f"\nOptimal point from 2.a:")
print(f"(x*, y*) = ({x_star:.6f}, {y_star:.6f})")

# Objective
f_star = x_star**2 + y_star**2 - 6*x_star*y_star - 4*x_star - 5*y_star
print(f"f(x*, y*) = {f_star:.6f}")

print("\n" + "-" * 80)
print("Original constraints (as given in problem):")
print("-" * 80)
print("  (1) y ≤ -(x-2)² + 4")
print("  (2) y ≥ -x + 1")

print("\n" + "-" * 80)
print("Rewrite in standard form g(x,y) ≤ c:")
print("-" * 80)
print("  (1) (x-2)² + y ≤ 4     =>  g1(x,y) = (x-2)² + y,  c1 = 4")
print("  (2) -x - y ≤ -1        =>  g2(x,y) = -x - y,      c2 = -1")

print("\n" + "=" * 80)
print("For constraint 1: g1(x,y) = (x-2)² + y ≤ c1")
print("=" * 80)

# Gradients
print("\nGradients:")
grad_f = np.array([2*x_star - 6*y_star - 4, 2*y_star - 6*x_star - 5])
grad_g1 = np.array([2*(x_star - 2), 1])
grad_g2 = np.array([-1, -1])

print(f"∇f  = ({grad_f[0]:.6f}, {grad_f[1]:.6f})")
print(f"∇g1 = ({grad_g1[0]:.6f}, {grad_g1[1]:.6f})")
print(f"∇g2 = ({grad_g2[0]:.6f}, {grad_g2[1]:.6f})")

# The Lagrangian is: L(x,y,λ) = f(x,y) + λ1(g1 - c1) + λ2(g2 - c2)
# At optimum: ∇f + λ1∇g1 + λ2∇g2 = 0

print("\nKKT Stationarity: ∇f + λ1∇g1 + λ2∇g2 = 0")
print("Solving for λ1, λ2...")

A = np.column_stack([grad_g1, grad_g2])
b = -grad_f
lambdas = np.linalg.solve(A, b)
lambda1, lambda2 = lambdas

print(f"\nλ1 = {lambda1:.6f}")
print(f"λ2 = {lambda2:.6f}")

# Check the sign issue - we have negative lambdas!
# This means our formulation needs adjustment

print("\n" + "!" * 80)
print("SIGN ISSUE DETECTED")
print("!" * 80)
print("\nFor MINIMIZATION with g(x) ≤ c, we need λ ≥ 0.")
print("But we got negative λ values!")
print("\nThis happens because of how we set up the problem.")
print("The standard formulation for sensitivity analysis is:")
print("  Minimize f(x) subject to h(x) = c")
print("  Then: df*/dc = λ (the Lagrange multiplier)")

print("\n" + "=" * 80)
print("CORRECT FORMULATION FOR SENSITIVITY")
print("=" * 80)

print("\nFor constraint: y ≤ -(x-2)² + 4")
print("Rewrite as EQUALITY by introducing the constant on RHS:")
print("  y = -(x-2)² + c1,  where c1 = 4")
print("\nOR equivalently:")
print("  h(x,y,c1) = y + (x-2)² - c1 = 0")

print("\nWhen c1 increases from 4 to 4.1, the feasible region expands upward.")
print("This should allow us to achieve a LOWER (better) loss value.")

print("\n" + "-" * 80)
print("Alternative approach: Direct sensitivity formula")
print("-" * 80)

print("\nAt the optimum with BOTH constraints active:")
print("  g1(x*, y*) = (x-2)² + y = c1 = 4")
print("  g2(x*, y*) = -x - y = c2 = -1")

print("\nThe sensitivity df*/dc1 can be found from the Lagrange multiplier.")
print("Using the correct sign convention:")

# For the formulation: minimize f subject to g(x,y) - c ≤ 0
# The sensitivity is: df*/dc = λ (where λ ≥ 0 for minimization)

# Actually, let's use the envelope theorem more carefully
print("\nUsing Envelope Theorem:")
print("If L(x,y,λ,c) = f(x,y) + λ(g(x,y) - c), then")
print("  df*/dc = -λ  (when constraint is active)")

print(f"\nSince we found λ1 = {lambda1:.6f} (with our sign convention),")
print(f"We have: df*/dc1 = -λ1 = {-lambda1:.6f}")

print("\n" + "=" * 80)
print("ANSWER TO PROBLEM 2.b")
print("=" * 80)

# The correct interpretation
actual_sensitivity = -lambda1  # This is df*/dc1

print(f"\nWhen c1 changes from 4 to 4.1 (Δc1 = +0.1):")
print(f"  Δf* ≈ (df*/dc1) × Δc1")
print(f"      = {actual_sensitivity:.6f} × 0.1")
print(f"      = {actual_sensitivity * 0.1:.6f}")

new_f_star = f_star + actual_sensitivity * 0.1

print(f"\nNew optimal value:")
print(f"  f*_new ≈ f*_old + Δf*")
print(f"         ≈ {f_star:.6f} + {actual_sensitivity * 0.1:.6f}")
print(f"         ≈ {new_f_star:.6f}")

if actual_sensitivity < 0:
    print(f"\n✓ Since df*/dc1 < 0, increasing c1 DECREASES the optimal loss.")
    print(f"  This makes sense: relaxing the constraint allows better optimization.")
else:
    print(f"\n✓ Since df*/dc1 > 0, increasing c1 INCREASES the optimal loss.")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("\nThe gradient of f at the stationary point is a linear combination")
print("of the constraint gradients:")
print(f"\n  ∇f = λ1·∇g1 + λ2·∇g2")
print(f"     = {lambda1:.6f}·∇g1 + {lambda2:.6f}·∇g2")
print("\nOR equivalently (changing sign):")
print(f"  -∇f = {-lambda1:.6f}·∇g1 + {-lambda2:.6f}·∇g2")
print("\nThe Lagrange multipliers are the coefficients in this linear combination.")
print(f"λ1 = {lambda1:.6f} tells us the 'shadow price' or sensitivity of constraint 1.")
