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
    diagonal[-1] = 1
    lower[-1] = -1

    return diagonal, lower, upper


def create_fast_edp_solver_robin_neumann(nb_space_points, space_step, time_step, diffusion_coefficient,
                                         transport_variable, nucleus_permeability):
    discrete_steps = time_step, space_step

    transport_values_over_space = compute_transport_values_over_space(nb_space_points, transport_variable)

    system_coefficients = diffusion_coefficient, transport_values_over_space, transport_variable, nucleus_permeability

    content_system_matrix = create_diags_for_system_matrix_robin_neumann(nb_space_points, discrete_steps,
                                                                         system_coefficients)

    system_matrix = create_sparse_matrix_from_diags(nb_space_points, content_system_matrix)

    fast_edp_solver = factorized(system_matrix)

    return fast_edp_solver


def compute_transport_values_over_space(nb_space_points, transport_variable):
    space_array = np.linspace(0, 1, nb_space_points)
    transport_constant = 10
    transport_values_over_space = transport_variable * np.exp(-transport_constant * space_array)
    transport_values_over_space[transport_values_over_space < 1e-2] = 0

    return transport_values_over_space


def create_diags_for_system_matrix_robin_neumann(nb_space_points, discrete_steps, coefficients):
    time_step, spatial_step = discrete_steps

    diffusion_coefficient, transport_values_over_space, transport_variable, nucleus_permeability = coefficients

    discrete_laplacian = time_step / spatial_step ** 2
    discrete_derivative = time_step / spatial_step

    diagonal = np.zeros(nb_space_points)
    lower = np.zeros(nb_space_points - 1)
    upper = np.zeros(nb_space_points - 1)

    diagonal[:] = 1 + 2 * diffusion_coefficient * discrete_laplacian + transport_values_over_space * discrete_derivative
    lower[:] = - discrete_laplacian * diffusion_coefficient
    upper[:] = - discrete_laplacian * diffusion_coefficient - transport_values_over_space[1:] * discrete_derivative

    # Robin-Neumann boundary conditions
    diagonal[0] = 1 + spatial_step * nucleus_permeability / diffusion_coefficient
    diagonal[-1] = -1
    lower[-1] = 1 - spatial_step * transport_values_over_space[-1] / diffusion_coefficient
    upper[0] = -(1 + spatial_step * transport_variable / diffusion_coefficient)

    return diagonal, lower, upper


def create_sparse_matrix_from_diags(nb_space_points, diagonals_content):
    system_matrix = sparse.diags(diagonals=diagonals_content, offsets=[0, -1, 1],
                                 shape=(nb_space_points, nb_space_points), format='csr')
    # Conversion to Compressed Sparse Column format for efficiency
    system_matrix = system_matrix.tocsc()
    return system_matrix


def create_reaction_array(system_values, parameters_values):
    monomers, dimers, apoe_proteins, complexes = system_values
    k, ka, fragmentation_for_complexes, fragmentation_for_dimers = parameters_values
    reaction_array = np.zeros_like(monomers)

    reaction_array[1:-1] = (-k * monomers[1:-1] * monomers[1:-1] - ka * monomers[1:-1] * apoe_proteins[1:-1]
                            + fragmentation_for_complexes * complexes[1:-1]
                            + fragmentation_for_dimers * 2 * dimers[1:-1])

    reaction_array[0] = -monomers[0]
    reaction_array[-1] = -monomers[-1]

    return reaction_array


if __name__ == "__main__":
    pass
