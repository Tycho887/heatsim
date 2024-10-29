import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
# Pipe dimensions
Lx, Ly = 5.0, 1.0     # Pipe length and width
nx, ny = 100, 20      # Grid points in x and y directions

# Discretization
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Physical properties
rho, nu = 10.0, 0.5    # Density and kinematic viscosity
U_inlet, g = 0.0, 10.0 # Inlet velocity and gravity
U_upper, U_lower = 0, -0  # Upper and lower wall velocities

# Time-stepping
dt, nt = 0.001, 1000   # Time step size and number of time steps

# --- Field Initialization ---
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # Pressure
b = np.zeros((ny, nx))  # RHS of the pressure-Poisson equation

# Boundary conditions
# Upper wall: rightward flow
u[-1, :] = U_upper
v[-1, :] = 0
    
# Lower wall: rotating leftward flow
u[0, :] = U_lower
v[0, :] = 0
    
# Side walls: no-slip
u[:, 0] = u[:, -1] = 0
v[:, 0] = v[:, -1] = 0

# --- Helper Functions ---
def build_up_b(b, u, v, rho, dt, dx, dy):
    """
    Build RHS for the pressure Poisson equation.
    """
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                    (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                         (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b, rho, g):
    """
    Solve the pressure Poisson equation with boundary conditions.
    """
    pn = np.empty_like(p)
    for _ in range(50):  # Gauss-Seidel iterations
        pn[:] = p
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Boundary conditions
        p[:, 0], p[:, -1], p[0, :], p[-1, :] = p[:, 1], p[:, -2], p[1, :], p[-2, :]
    
    p[:, :] += rho * g * np.linspace(0, -Ly, ny).reshape(-1, 1)  # Gravity term
    p[:, 0] = 1  # Inlet pressure
    return p

def update_velocities(u, v, un, vn, p, rho, nu, dt, dx, dy, g):
    """
    Update the velocity fields u and v.
    """
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
                           dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])) -
                     g * dt)

    # # Boundary conditions
    # u[:, 0], u[:, -1] = U_inlet, u[:, -2]
    # u[0, :], u[-1, :], v[0, :], v[-1, :] = 0, 0, 0, 0

    # # Change boundary conditions to force a rotating flow
    # u[:, 0], u[:, -1] = 0, 0, 0, 0
    # u[0, :], u[-1, :], v[0, :], v[-1, :] = U_inlet, -U_inlet

    return u, v

# Boundary conditions
def apply_boundary_conditions(u, v, U_upper=1.0, U_lower=-1.0):
    """
    Apply custom boundary conditions:
    - Upper wall: flow to the right (positive u velocity)
    - Lower wall: rotating flow to the left (negative u velocity)
    - Side walls: no-slip conditions (u and v = 0)
    """
    # Upper wall: rightward flow
    u[-1, :] = U_upper
    v[-1, :] = 0
    
    # Lower wall: rotating leftward flow
    u[0, :] = U_lower
    v[0, :] = 0
    
    # Side walls: no-slip
    u[:, 0] = u[:, -1] = 0
    v[:, 0] = v[:, -1] = 0
    return u, v


# --- Main Simulation Loop ---
for n in range(nt):
    un, vn = u.copy(), v.copy()
    b = build_up_b(b, u, v, rho, dt, dx, dy)
    p = pressure_poisson(p, dx, dy, b, rho, g)
    u, v = update_velocities(u, v, un, vn, p, rho, nu, dt, dx, dy, g)
    u, v = apply_boundary_conditions(u, v)

# --- Visualization ---
fig, ax = plt.subplots()
x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)

def animate(i):
    ax.clear()
    ax.quiver(x, y, u, v)
    ax.set_title(f"Time step: {i}")

ani = animation.FuncAnimation(fig, animate, frames=range(0, nt, nt // 100), repeat=False)
plt.show()
