import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 1.0          # Length of the domain (meters)
T = 1.0          # Total time (seconds)
nx = 50          # Number of spatial cells
nt = 500         # Number of time steps
alpha = 0.05     # Thermal diffusivity (m^2/s)

# Derived parameters
dx = L / nx      # Spatial step size
dt = T / nt      # Time step size

# Stability criterion for explicit scheme
assert dt <= dx**2 / (2 * alpha), "Time step is too large for stability."

# Initialize the temperature field
u = np.zeros(nx + 2)         # Solution at the current time step (including ghost cells)
u_new = np.zeros(nx + 2)     # Solution at the next time step

# Initial condition (e.g., initial temperature distribution)
u[:] = 0.0

# Arrays to store results for visualization
results = [u.copy()]

# Time integration using FVM
for n in range(nt):
    # Update each internal cell using FVM
    for i in range(1, nx + 1):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1])

    # Apply boundary conditions (Dirichlet: fixed temperature at boundaries)
    u_new[0] = 100.0
    u_new[-1] = 50.0

    # Swap arrays
    u[:] = u_new
    results.append(u.copy())

# Visualization
fig, ax = plt.subplots()
x = np.linspace(0, L, nx + 2)

def animate(i):
    ax.clear()
    ax.set_ylim(0, 110)
    ax.plot(x, results[i], color='b')
    ax.set_title(f"Time step: {i}")

ani = animation.FuncAnimation(fig, animate, frames=range(0, nt, nt // 100), repeat=False)
plt.show()
