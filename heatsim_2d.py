import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters

Lx = 1.0          # Length of the domain in x-direction (meters)
Ly = 1.0          # Length of the domain in y-direction (meters)

T = 1.0          # Total time (seconds)
nx = 50          # Number of spatial cells in x-direction
ny = 50          # Number of spatial cells in y-direction
nt = 500         # Number of time steps

alpha = 0.05     # Thermal diffusivity (m^2/s)

# Derived parameters

dx = Lx / nx      # Spatial step size in x-direction
dy = Ly / ny      # Spatial step size in y-direction
dt = T / nt      # Time step size

# Stability criterion for explicit scheme

assert dt <= dx**2 / (4 * alpha), "Time step is too large for stability."

# Initialize the temperature field as a random distribution

u = np.zeros((nx + 2, ny + 2))

# Boundary conditions (Dirichlet: fixed temperature at boundaries)

# left_wall = 100.0
# right_wall = 50.0
# bottom_wall = 700.0
# top_wall = 700

# Arrays to store results for visualization

results = [u.copy()]

# Time integration using FVM

for n in range(nt):
    # Update each internal cell using FVM
    for i in range(0, nx + 1):
        for j in range(0, ny + 1):
            u[i, j] = u[i, j] + alpha * dt / dx**2 * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) + alpha * dt / dy**2 * (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])

    # Apply boundary conditions (Dirichlet: fixed temperature at boundaries)
    # u[0, :] = left_wall
    u[25, 25] = 100.0

    results.append(u.copy())

# Visualization

fig, ax = plt.subplots()
x = np.linspace(0, Lx, nx + 2)
y = np.linspace(0, Ly, ny + 2)

def animate(i):
    ax.clear()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.imshow(results[i], cmap='hot', origin='lower', extent=[0, Lx, 0, Ly])
    ax.set_title(f"Time step: {i}")

ani = animation.FuncAnimation(fig, animate, frames=range(0, nt, nt // 100), repeat=False)

plt.show()



