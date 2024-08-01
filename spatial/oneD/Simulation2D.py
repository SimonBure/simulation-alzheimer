import numpy as np
import scipy.sparse
from OneDimSpace import OneDimeSpace


class Space2D:
    def __init__(self, x: OneDimeSpace, y: OneDimeSpace):
        self.x = x
        self.y = y


class Simulation2D:
    def __init__(self, spatial_space: Space2D, time_space: OneDimeSpace, theta: float):
        self.spatial_space = spatial_space
        self.nx = spatial_space.x.nb_points
        self.ny = spatial_space.y.nb_points
        self.nb_points_diag = (self.nx + 1) * (self.ny + 1)

        self.time_space = time_space

        self.theta = theta

        self.diffusion_array = np.zeros(self.nb_points_diag)


    def compute_sparse_diffusion_matrix(self):
        diagonal = np.zeros(self.nb_points_diag)
        lower_diag = np.zeros(self.nb_points_diag - 1)
        upper_diag = np.zeros(self.nb_points_diag - 1)
        lowest_diag = np.zeros(self.nb_points_diag - (self.nx + 1))  # lower diagonal
        highest_diag = np.zeros(self.nb_points_diag - (self.nx + 1))  # upper diagonal

        fx = self.compute_fourier_number_x()
        fy = self.compute_fourier_number_y()

        m = self.map_indexes_to_single_number

        lower_offset = 1
        lowest_offset = self.nx + 1

        j = 0
        diagonal[m(0, j):m(self.nx + 1, j)] = 1  # j=0 boundary line
        for j in range(1, self.ny - 1):
            i = 0
            diagonal[m(i, j)] = 1  # Boundary
            i = self.spatial_space.x.nb_points
            diagonal[m(i, j)] = 1  # Boundary

            lowest_diag[m(1, j) - lowest_offset:m(self.nx, j) - lowest_offset] = - self.theta * fy
            lower_diag[m(1, j) - lower_offset:m(self.nx, j) - lower_offset] = - self.theta * fx
            diagonal[m(1, j):m(self.nx, j)] = 1 + 2 * self.theta * (fx + fy)
            upper_diag[m(1, j):m(self.nx, j)] = - self.theta * fx
            highest_diag[m(1, j):m(self.nx, j)] = - self.theta * fy

        j = self.ny
        diagonal[m(0, j):m(self.nx + 1, j)] = 1  # Boundary line

        sparse_diffusion_matrix = scipy.sparse.diags([diagonal, lower_diag, upper_diag, lowest_diag, highest_diag],
                                                     [0, -lower_offset, lower_offset, -lowest_offset, lowest_offset],
                                                     shape=(self.nb_points_diag, self.nb_points_diag), format='csr')
        return sparse_diffusion_matrix

    def compute_fourier_number_x(self):
        return self.time_space.step / (self.spatial_space.x.step ** 2)

    def compute_fourier_number_y(self):
        return self.time_space.step / (self.spatial_space.y.step ** 2)

    def map_indexes_to_single_number(self, i, j):
        return j * (self.nx + 1) + i

    def simulate(self):
        u = np.zeros((self.nx + 1, self.ny + 1))  # unknown u at new time level
        u_n = np.zeros((self.nx + 1, self.ny + 1))  # u at the previous time level
        b = np.zeros(self.nb_points_diag)  # right-hand side

        fx = self.compute_fourier_number_x()
        fy = self.compute_fourier_number_y()

        m = self.map_indexes_to_single_number  # for code density

        sparse_diffusion_matrix = self.compute_sparse_diffusion_matrix()

        for t in range(self.time_space.nb_points):
            f_a_np1 = np.zeros((self.spatial_space.x.nb_points, self.spatial_space.y.nb_points))
            f_a_n = np.zeros((self.spatial_space.x.nb_points, self.spatial_space.y.nb_points))

            self.fill_line_boundary(0)
            self.fill_line_boundary(self.ny)

            for j in range(1, self.spatial_space.y.nb_points):
                self.fill_index_boundary(j)



                self.diffusion_array[m(i_min, j):m(i_max, j)] =

    def fill_line_boundary(self, line_index: int):
        m = self.map_indexes_to_single_number  # for code density
        self.diffusion_array[m(0, line_index):m(self.nx + 1, line_index)] = 0

    def fill_index_boundary(self, index: int):
        self.diffusion_array[self.map_indexes_to_single_number(0, index)] = 0
        self.diffusion_array[self.map_indexes_to_single_number(self.nx, index)] = 0

    def update_unknown(self, unknown_actual):
        i_min = self.spatial_space.x.space[1]
        i_max = self.spatial_space.x.space[-1]

        unknown_actual[i_min:i_max, j] + (1 - theta) * (Fx * (unknown_actual[i_min + 1:i_max + 1, j] - 2 * unknown_actual[i_min:i_max, j] + unknown_actual[i_min - 1:i_max - 1, j]) + Fy * (unknown_actual[i_min:i_max, j + 1] - 2 * unknown_actual[i_min:i_max, j] + unknown_actual[i_min:i_max, j - 1])) + theta * dt * f_a_np1[i_min:i_max, j] + (1 - theta) * dt * f_a_n[i_min:i_max, j]

if __name__ == "__main__":
    a_time_space = OneDimeSpace(50, 1000, 1e-3)
    a_x_space = OneDimeSpace(1, 1000, 1e-2)
    a_y_space = OneDimeSpace(1, 1000, 1e-2)
    a_2d_spatial_space = Space2D(a_x_space, a_y_space)
    a_theta = 1  # forward Euler numerical scheme
    a_2d_sim = Simulation2D(a_2d_spatial_space, a_time_space, a_theta)
