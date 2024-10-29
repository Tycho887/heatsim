import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 41, 41  # Grid points
nt = 500  # Number of time steps
dt = 0.01  # Time step
Lx, Ly = 2.0, 2.0  # Domain length in x and y directions
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Grid spacing
rho = 1.0  # Density
nu = 0.1  # Kinematic viscosity

# Initialize fields
u = np.zeros((ny, nx))  # Velocity in x-direction
v = np.zeros((ny, nx))  # Velocity in y-direction
p = np.zeros((ny, nx))  # Pressure

# Function to calculate the RHS of the pressure Poisson equation
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(50):  # Iterative solver loop
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) / (2 * (dx**2 + dy**2)) - \
                        b[1:-1, 1:-1] * dx**2 * dy**2 / (2 * (dx**2 + dy**2))
        
        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = Lx
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = Ly
    return p

# Solver
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute source term for the pressure Poisson equation
    b = (rho * ((1 / dt) * ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) + 
                             (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)) -
                ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx))**2 -
                2 * ((un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy) *
                     (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)) -
                ((vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy))**2))
    
    # Solve for pressure
    p = pressure_poisson(p, dx, dy, b)
    
    # Update velocity field using Navier-Stokes equations
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
    
    # Apply boundary conditions for velocity
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0

# Plot results
plt.imshow(p, cmap="viridis")
plt.colorbar()
plt.title("Pressure field")
plt.show()
