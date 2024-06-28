import numpy as np
import scipy.sparse as sparse
from matplotlib import pyplot as plt
from scipy.sparse.linalg import factorized


def spatial_simulation():
    spatial_array = np.linspace(start=0, stop=10, num=100)


def create_fast_edp_solver_neumann(nb_space_points, space_step, time_step, diffusion_coefficient):
    diagonal = np.zeros(nb_space_points)
    lower = np.zeros(nb_space_points - 1)
    upper = np.zeros(nb_space_points - 1)

    diffusion_term = (time_step / space_step ** 2)

    diagonal[:] = 1 + 2 * diffusion_coefficient * diffusion_term
    lower[:] = - diffusion_term * diffusion_coefficient
    upper[:] = - diffusion_term * diffusion_coefficient

    # TO-DO create a function to setup boundary conditions
    # Neumann Boundary conditions
    diagonal[0] = 1
    upper[0] = -1
    diagonal[nb_space_points - 1] = 1
    lower[-1] = -1

    system_matrix = create_matrix_edp_system(nb_space_points, diagonal, lower, upper)

    return factorized(system_matrix)


def create_fast_edp_solver_robin_neumann(nb_space_points, space_step, time_step, diffusion_coefficient,
                                         transport_variable, alpha):
    diagonal = np.zeros(nb_space_points)
    lower = np.zeros(nb_space_points - 1)
    upper = np.zeros(nb_space_points - 1)

    # TO-DO find better names for all the transport things
    xx_vec = np.linspace(0, 1, Nx)
    transport_constant = 10
    transport_coefficient = transport_variable * np.exp(-transport_constant * xx_vec)
    transport_coefficient[transport_coefficient < 1e-2] = 0

    transport_term = transport_coefficient * time_step / space_step

    diffusion_term = (time_step / space_step ** 2)

    diagonal[:] = 1 + 2 * diffusion_coefficient * diffusion_term + transport_term
    lower[:] = - diffusion_term * diffusion_coefficient
    upper[:] = - diffusion_term * diffusion_coefficient - transport_term[1:]

    # TO-DO create a function to setup boundary conditions
    # Robin-Neumann boundary conditions
    ##Attention on sait pas si
    diagonal[0] = 1 + dx * alpha / D
    upper[0] = -(1 + dx * v / D)
    # ou
    # diagonal[0] = -1
    # upper[0] = 1+dx*(v-alpha)/D
    diagonal[Nx - 1] = -1
    lower[-1] = 1 - dx * vv_vec[-1] / D
    # lower[-1] = 1

    system_matrix = create_matrix_edp_system(nb_space_points, diagonal, lower, upper)

    return factorized(system_matrix)


# TO-DO regroup diagonal, lower, upper in one parameter
def create_matrix_edp_system(nb_space_points, diagonal, lower, upper):
    system_matrix = spsp.diags(diagonals=[diagonal, lower, upper], offsets=[0, -1, 1],
                               shape=(nb_space_points, nb_space_points), format='csr')
    # Conversion to Compressed Sparse Column format for efficiency
    system_matrix = system_matrix.tocsc()
    return system_matrix
