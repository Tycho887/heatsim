import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Pipe dimensions
Lx, Ly = 5.0, 1.0       # Length and width of the pipe
nx, ny = 100, 20          # Number of grid points in x and y directions

# Discretization
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Physical properties
rho = 10.0               # Density
nu = 0.5                 # Kinematic viscosity
U_inlet = .0            # Inlet velocity
g = 10              # Acceleration due to gravity

# Time-stepping parameters
dt = 0.001               # Time step size
nt = 1000                # Number of time steps

# Initialize fields
u = np.zeros((ny, nx))   # x-velocity
v = np.zeros((ny, nx))   # y-velocity
p = np.zeros((ny, nx))   # Pressure
b = np.zeros((ny, nx))   # RHS of the pressure-Poisson equation

# Boundary conditions
u[:, 0] = U_inlet        # Inlet velocity
u[:, -1] = u[:, -2]      # Outlet (zero-gradient)
v[:, 0] = v[:, -1] = 0   # Inlet and outlet v-velocity
u[0, :] = u[-1, :] = 0   # No-slip at walls
v[0, :] = v[-1, :] = 0   # No-slip at walls

# Helper functions
def build_up_b(b, u, v, rho, dt, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) + 
                     (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                         (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for _ in range(50):  # Perform a few Gauss-Seidel iterations
        pn[:] = p
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        
        # Apply boundary conditions
        p[:, -1] = p[:, -2]  # Outlet: zero-gradient
        p[:, 0] = p[:, 1]    # Inlet: zero-gradient
        p[0, :] = p[1, :]    # Top wall: zero-gradient
        p[-1, :] = p[-2, :]  # Bottom wall: zero-gradient

    p[:, :] += rho * g * np.linspace(0, -Ly, ny).reshape(-1, 1)

    # Inlet presure 101325 Pa

    p[:, 0] = 1
        
    return p

# Main time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Compute the RHS of the pressure Poisson equation
    b = build_up_b(b, u, v, rho, dt, dx, dy)

    # Solve for pressure
    p = pressure_poisson(p, dx, dy, b)

    # Update velocities
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

    # Add gravity term in the vertical (v) direction
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])) -
                     g * dt)  # Gravity term added here

    # Apply boundary conditions
    u[:, 0] = U_inlet
    u[:, -1] = u[:, -2]
    u[0, :] = u[-1, :] = 0
    v[:, 0] = v[:, -1] = 0
    v[0, :] = v[-1, :] = 0

# Final result: u, v, and p contain the velocity and pressure fields

fig, ax = plt.subplots()
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# CS = ax.contourf(x, y, np.sqrt(u**2 + v**2), alpha=0.5)
CS = ax.contourf(x, y, np.sqrt(u**2), alpha=0.5)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('Color scale') 

# Plot the velocity field

def animate(i):
    ax.clear()
    ax.quiver(x, y, u, v)
    ax.set_title(f"Time step: {i}")

# Plot absolute velocity

# def animate(i):
#     ax.clear()
#     ax.contourf(x, y, np.sqrt(u**2 + v**2), alpha=0.5)
#     ax.set_title(f"Time step: {i}")
#     # plot scale for the colorbar


# # Plot the pressure field

# def animate(i):
#     ax.clear()
#     ax.contourf(x, y, p, alpha=0.5)
#     ax.set_title(f"Time step: {i}")

# Plot numerical error

ani = animation.FuncAnimation(fig, animate, frames=range(0, nt, nt // 100), repeat=False)

plt.show()