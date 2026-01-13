import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = 2x1^2 - 1.05x1^4 + (1/6)x1^6 - x1*x2 + x2^2
def f(x1, x2):
    return 2*x1**2 - 1.05*x1**4 + (1/6)*x1**6 - x1*x2 + x2**2

# Create grid for contour plot
x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Create the contour plot
plt.figure(figsize=(10, 8))
contour = plt.contour(X1, X2, Z, levels=30, colors='black', linewidths=0.5)
contourf = plt.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.7)

# Add colorbar
cbar = plt.colorbar(contourf)
cbar.set_label('f(x)', rotation=270, labelpad=20, fontsize=12)

# Label contour lines
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

# Find and plot critical points (stationary points)
# Global minima are approximately at:
# (0, 0), (-1.0285, -0.5142), (1.0285, 0.5142)
critical_points = [
    (0, 0),
    (-1.0285, -0.5142),
    (1.0285, 0.5142)
]

for point in critical_points:
    plt.plot(point[0], point[1], 'r*', markersize=15, label='Critical point' if point == critical_points[0] else '')

# Labels and title
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title(r'Contour Plot of $f(x) = 2x_1^2 - 1.05x_1^4 + \frac{1}{6}x_1^6 - x_1x_2 + x_2^2$', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Save the plot
plt.tight_layout()
plt.savefig('/Users/ruijiang/Programming/my-git/rjiang6-upenn/ese5460/Assignments/HW0/contour_plot.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
print("Contour plot saved as contour_plot.pdf")

# Also create a 3D surface plot for better visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Coarser grid for 3D plot (for performance)
x1_3d = np.linspace(-3, 3, 100)
x2_3d = np.linspace(-3, 3, 100)
X1_3d, X2_3d = np.meshgrid(x1_3d, x2_3d)
Z_3d = f(X1_3d, X2_3d)

# Plot surface
surf = ax.plot_surface(X1_3d, X2_3d, Z_3d, cmap='viridis', alpha=0.8,
                       linewidth=0, antialiased=True)

# Plot critical points
for point in critical_points:
    z_val = f(point[0], point[1])
    ax.scatter(point[0], point[1], z_val, c='red', s=100, marker='*',
              edgecolors='black', linewidth=1)

# Labels
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_zlabel('$f(x)$', fontsize=12)
ax.set_title(r'3D Surface Plot of $f(x) = 2x_1^2 - 1.05x_1^4 + \frac{1}{6}x_1^6 - x_1x_2 + x_2^2$', fontsize=11)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.savefig('/Users/ruijiang/Programming/my-git/rjiang6-upenn/ese5460/Assignments/HW0/surface_plot.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
print("3D surface plot saved as surface_plot.pdf")

plt.show()

print("\nCritical points (stationary points):")
for i, point in enumerate(critical_points):
    f_val = f(point[0], point[1])
    print(f"Point {i+1}: x1 = {point[0]:.4f}, x2 = {point[1]:.4f}, f(x) = {f_val:.6f}")
