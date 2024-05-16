import matplotlib.pyplot as plt
import numpy as np

# Define the flexible hatch pattern
def flexible_hatch(x, hatch_distance=10, hatch_thickness=2):
    return [(0, 0), (hatch_distance, hatch_distance)], [(0, hatch_thickness), (hatch_thickness, 0)]

# Create the figure and axis
fig, ax = plt.subplots()

# Define the x values
x = np.linspace(-4, 4, 100)

# Plot the function z
ax.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2), color='red', label='z')

# Plot the filled areas with flexible hatch pattern
ax.fill_between(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2), where=(x >= 0) & (x <= 1), color='none', hatch=flexible_hatch(x), edgecolor='red', label='Interval 1')
ax.fill_between(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2), where=(x >= -2) & (x <= -0.5), color='cyan', hatch=flexible_hatch(x, hatch_distance=5, hatch_thickness=0.5), edgecolor='blue', label='Interval 2')

# Set axis limits and labels
ax.set_xlim(-4, 4)
ax.set_xlabel('z')
ax.set_ylim(0, 1)

# Show legend
ax.legend()

plt.show()



