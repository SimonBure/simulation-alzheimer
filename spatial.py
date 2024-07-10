import numpy as np
import scipy.sparse as sparse
from matplotlib import pyplot as plt
from scipy.sparse.linalg import factorized


def spatial_simulation():
    spatial_array = np.linspace(start=0, stop=10, num=100)


def create_fast_edp_solver_neumann(nb_space_points, space_step, time_step, diffusion_coefficient):
    discrete_steps = time_step, space_step

    content_system_matrix = create_diags_for_system_matrix_neumann(nb_space_points, discrete_steps,
                                                                   diffusion_coefficient)

    system_matrix = create_sparse_matrix_from_diags(nb_space_points, content_system_matrix)

    fast_edp_solver = factorized(system_matrix)

    return fast_edp_solver


def create_diags_for_system_matrix_neumann(nb_space_points, discrete_steps, diffusion_coefficient):
    time_step, space_step = discrete_steps
    discrete_laplacian = time_step / space_step**2

    diagonal = np.zeros(nb_space_points)
    lower = np.zeros(nb_space_points - 1)
    upper = np.zeros(nb_space_points - 1)

    diagonal[:] = 1 + 2 * diffusion_coefficient * discrete_laplacian
    lower[:] = - discrete_laplacian * diffusion_coefficient
    upper[:] = - discrete_laplacian * diffusion_coefficient

    # Neumann boundary conditions
    diagonal[0] = 1
    upper[0] = -1
    diagonal[nb_space_points - 1] = 1
    lower[-1] = -1

    return diagonal, lower, upper


def create_fast_edp_solver_robin_neumann(nb_space_points, space_step, time_step, diffusion_coefficient,
                                         transport_variable, alpha):
    discrete_steps = time_step, space_step

    transport_coefficient = compute_transport_coefficient(nb_space_points, transport_variable)

    system_coefficients = diffusion_coefficient, transport_coefficient, transport_variable, alpha

    content_system_matrix = create_diags_for_system_matrix_robin_neumann(nb_space_points, discrete_steps,
                                                                         system_coefficients)

    system_matrix = create_sparse_matrix_from_diags(nb_space_points, content_system_matrix)

    fast_edp_solver = factorized(system_matrix)

    return fast_edp_solver


def compute_transport_coefficient(nb_space_points, transport_variable):
    space_array = np.linspace(0, 1, nb_space_points)
    transport_constant = 10
    transport_coefficient = transport_variable * np.exp(-transport_constant * space_array)
    transport_coefficient[transport_coefficient < 1e-2] = 0

    return transport_coefficient


def create_diags_for_system_matrix_robin_neumann(nb_space_points, discrete_steps, coefficients):
    time_step, spatial_step = discrete_steps

    discrete_laplacian = time_step / space_step ** 2
    discrete_derivative = time_step / space_step

    diffusion_coefficient, transport_coefficient, transport_variable, alpha = coefficients

    diagonal = np.zeros(nb_space_points)
    lower = np.zeros(nb_space_points - 1)
    upper = np.zeros(nb_space_points - 1)

    diagonal[:] = 1 + 2 * diffusion_coefficient * discrete_laplacian + transport_coefficient * discrete_derivative
    lower[:] = - discrete_laplacian * diffusion_coefficient
    upper[:] = - discrete_laplacian * diffusion_coefficient - transport_coefficient[1:] * discrete_derivative[1:]

    # Robin-Neumann boundary conditions
    diagonal[0] = 1 + spatial_step * alpha / diffusion_coefficient
    upper[0] = -(1 + spatial_step * transport_variable / diffusion_coefficient)
    diagonal[nb_space_points - 1] = -1
    lower[-1] = 1 - spatial_step * transport_coefficient[-1] / diffusion_coefficient

    return diagonal, lower, upper


def create_sparse_matrix_from_diags(nb_space_points, diagonals_content):
    system_matrix = sparse.diags(diagonals=diagonals_content, offsets=[0, -1, 1],
                                 shape=(nb_space_points, nb_space_points), format='csr')
    # Conversion to Compressed Sparse Column format for efficiency
    system_matrix = system_matrix.tocsc()
    return system_matrix


def Reaction_P1_Robin(p1, p2, p1a, p2a, k, ka, tauC, tauD):
    res = np.zeros_like(p1);

    res[1:-1] = -k * p1[1:-1] * p1[1:-1] - ka * p1[1:-1] * p1a[1:-1] + tauD * 2 * p2[1:-1] + tauC * p2a[1:-1];

    res[0] = -p1[0];
    res[-1] = -p1[-1];

    return (res)


if __name__ == "__main__":
    pass
