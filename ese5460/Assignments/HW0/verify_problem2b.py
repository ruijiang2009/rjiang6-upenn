import numpy as np
import sympy as sp

print("=" * 60)
print("Problem 2.a: Finding Optimal Point")
print("=" * 60)

# Define the problem symbolically
x, y, lam1, lam2 = sp.symbols('x y lambda1 lambda2', real=True)

# Objective function: f(x,y) = x^2 + y^2 - 6xy - 4x - 5y
f = x**2 + y**2 - 6*x*y - 4*x - 5*y

# Constraints (in form g <= 0):
# g1: y + (x-2)^2 - 4 <= 0  => y <= -(x-2)^2 + 4
# g2: -x + 1 - y <= 0       => y >= -x + 1
g1 = y + (x-2)**2 - 4
g2 = -x + 1 - y

# Gradients
grad_f = [sp.diff(f, x), sp.diff(f, y)]
grad_g1 = [sp.diff(g1, x), sp.diff(g1, y)]
grad_g2 = [sp.diff(g2, x), sp.diff(g2, y)]

print("\nGradients:")
print(f"∇f = ({grad_f[0]}, {grad_f[1]})")
print(f"∇g1 = ({grad_g1[0]}, {grad_g1[1]})")
print(f"∇g2 = ({grad_g2[0]}, {grad_g2[1]})")

# KKT conditions (assuming both constraints are active):
# ∇f + λ1∇g1 + λ2∇g2 = 0
# g1 = 0 (constraint 1 active)
# g2 = 0 (constraint 2 active)

eq1 = grad_f[0] + lam1*grad_g1[0] + lam2*grad_g2[0]  # ∂L/∂x = 0
eq2 = grad_f[1] + lam1*grad_g1[1] + lam2*grad_g2[1]  # ∂L/∂y = 0
eq3 = g1  # g1 = 0 (active)
eq4 = g2  # g2 = 0 (active)

print("\nKKT System (both constraints active):")
print(f"eq1: {eq1} = 0")
print(f"eq2: {eq2} = 0")
print(f"eq3: {eq3} = 0")
print(f"eq4: {eq4} = 0")

# Solve the system
solutions = sp.solve([eq1, eq2, eq3, eq4], [x, y, lam1, lam2])

print("\n" + "=" * 60)
print("Solutions:")
print("=" * 60)

for i, sol in enumerate(solutions):
    print(f"\nSolution {i+1}:")
    x_val = float(sol[0])
    y_val = float(sol[1])
    lam1_val = float(sol[2])
    lam2_val = float(sol[3])

    print(f"  x = {sol[0]} ≈ {x_val:.6f}")
    print(f"  y = {sol[1]} ≈ {y_val:.6f}")
    print(f"  λ1 = {sol[2]} ≈ {lam1_val:.6f}")
    print(f"  λ2 = {sol[3]} ≈ {lam2_val:.6f}")

    # Calculate f at this point
    f_val = float(f.subs([(x, sol[0]), (y, sol[1])]))
    print(f"  f(x,y) = {f_val:.6f}")

    # Check if λ1, λ2 >= 0 (KKT conditions for minimization)
    if lam1_val >= 0 and lam2_val >= 0:
        print(f"  ✓ Valid (λ1, λ2 ≥ 0)")
    else:
        print(f"  ✗ Invalid (need λ1, λ2 ≥ 0 for minimum)")

print("\n" + "=" * 60)
print("Verification of the given answer:")
print("=" * 60)

x_star = (5 - np.sqrt(21)) / 2
y_star = (-3 + np.sqrt(21)) / 2

print(f"Given: (x*, y*) = ((5-√21)/2, (-3+√21)/2)")
print(f"       ≈ ({x_star:.6f}, {y_star:.6f})")

# Evaluate f at this point
f_star = x_star**2 + y_star**2 - 6*x_star*y_star - 4*x_star - 5*y_star
print(f"f(x*, y*) = {f_star:.6f}")

# Check constraints
g1_val = y_star + (x_star - 2)**2 - 4
g2_val = -x_star + 1 - y_star
print(f"\nConstraint checks:")
print(f"g1(x*, y*) = {g1_val:.10f} (should be ≈ 0)")
print(f"g2(x*, y*) = {g2_val:.10f} (should be ≈ 0)")

# Calculate gradients at this point
grad_f_val = np.array([2*x_star - 6*y_star - 4, 2*y_star - 6*x_star - 5])
grad_g1_val = np.array([2*(x_star - 2), 1])
grad_g2_val = np.array([-1, -1])

print(f"\nGradients at (x*, y*):")
print(f"∇f = ({grad_f_val[0]:.6f}, {grad_f_val[1]:.6f})")
print(f"∇g1 = ({grad_g1_val[0]:.6f}, {grad_g1_val[1]:.6f})")
print(f"∇g2 = ({grad_g2_val[0]:.6f}, {grad_g2_val[1]:.6f})")

# Find lambda values
# ∇f + λ1∇g1 + λ2∇g2 = 0
# This is a system: A * [λ1, λ2]^T = -∇f
A = np.column_stack([grad_g1_val, grad_g2_val])
lambdas = np.linalg.solve(A, -grad_f_val)

print(f"\nLagrange multipliers:")
print(f"λ1 = {lambdas[0]:.6f}")
print(f"λ2 = {lambdas[1]:.6f}")

print("\n" + "=" * 60)
print("Problem 2.b: Sensitivity Analysis")
print("=" * 60)

print(f"\nChange in constraint: Δc1 = 0.1")
print(f"Using λ1 = {lambdas[0]:.6f}")
print(f"\nEstimated change in optimal value:")
print(f"Δf* ≈ -λ1 * Δc1 = -{lambdas[0]:.6f} * 0.1 = {-lambdas[0] * 0.1:.6f}")
print(f"\nNew optimal value estimate:")
print(f"f*_new ≈ f*_old + Δf* = {f_star:.6f} + {-lambdas[0] * 0.1:.6f} = {f_star - lambdas[0] * 0.1:.6f}")
