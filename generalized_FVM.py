import numpy as np

# Assuming a rectangular domain with uniform grid spacing, described as a numpy array

class grid:
    def __init__(self, shape, size,boundary_conditions):
        self._domain = np.zeros(shape)
        self._size = np.array(size) # size of the domain in meters

        self._dy = self.size[0]/self.domain.shape[0]
        self._dx = self.size[1]/self.domain.shape[1]

        # boundary conditions describe the (left,right,top,bottom) dirichlet boundary conditions can be of form (1, None, 0, None)

        self._boundary_conditions = {"left": boundary_conditions[0], "right": boundary_conditions[1], "top": boundary_conditions[2], "bottom": boundary_conditions[3]}

    def apply_boundary_conditions(self):
        if self.boundary_conditions["left"] is not None:
            self.domain[:,0] = self.boundary_conditions["left"]
        if self.boundary_conditions["right"] is not None:
            self.domain[:,-1] = self.boundary_conditions["right"]
        if self.boundary_conditions["top"] is not None:
            self.domain[0,:] = self.boundary_conditions["top"]
        if self.boundary_conditions["bottom"] is not None:
            self.domain[-1,:] = self.boundary_conditions["bottom"]

    @property
    def domain(self):
        return self._domain
    
    @property
    def size(self):
        return self._size
    
    @property
    def dy(self):
        return self._dy
    
    @property
    def dx(self):
        return self._dx


class FVM:
    def __init__(self, shape, size, boundary_conditions, alpha, dt, nt):
        self.grid = grid(shape, size, boundary_conditions)
        # Alpha is the thermal diffusivity
        self.alpha = alpha

        self.dtdy2 = self.alpha * dt / self.grid.dy**2
        self.dtdx2 = self.alpha * dt / self.grid.dx**2

        self.nt = nt

    def time_integration(self, nt):
        pass


    def animate(self, variable):
        # implement this
        pass

    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        results = [self.grid.domain.copy()]
        fig, ax = plt.subplots()
        x = np.linspace(0, self.grid.size[1], self.grid.domain.shape[1])
        y = np.linspace(0, self.grid.size[0], self.grid.domain.shape[0])

        ani = animation.FuncAnimation(fig, self.animate, frames=range(0, self.nt, self.nt // 100), repeat=False)
        plt.show()