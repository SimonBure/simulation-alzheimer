import spatial as sp
import numpy as np


def test_create_sparse_matrix_from_diags():
    a_nb_space_points = 5

    some_diagonal = [1, 2, 3, 4, 5]
    some_upper_diagonal = [6, 7, 8, 9]
    some_lower_diagonal = [10, 11, 12, 13]
    some_diagonals_content = (some_diagonal, some_lower_diagonal, some_upper_diagonal)

    expected_output = np.array([[1, 6, 0, 0, 0],
                               [10, 2, 7, 0, 0],
                               [0, 11, 3, 8, 0],
                               [0, 0, 12, 4, 9],
                               [0, 0, 0, 13, 5]])

    actual_output = sp.create_sparse_matrix_from_diags(a_nb_space_points, some_diagonals_content).toarray()

    assert np.array_equal(expected_output, actual_output)


def test_compute_transport_values_over_space():
    a_nb_space_points = 5
    a_transport_variable = [1, 2, 3, 4, 5]

    expected_output = [1, 2 * np.exp(-2.5), 3 * np.exp(-5), 0, 0]

    actual_output = sp.compute_transport_values_over_space(a_nb_space_points, a_transport_variable)

    assert np.array_equal(expected_output, actual_output)


def test_create_diags_for_system_matrix_neumann():
    a_nb_space_points = 5
    some_discrete_steps = (0.1, 0.2)
    a_diffusion_coefficient = 3

    expected_diagonal = [1.0, 16.0, 16.0, 16.0, 1.0]
    expected_lower_diagonal = [-7.5, -7.5, -7.5, -1]
    expected_upper_diagonal = [-1, -7.5, -7.5, -7.5]

    actual_diagonal, actual_lower, actual_upper = sp.create_diags_for_system_matrix_neumann(a_nb_space_points,
                                                                                            some_discrete_steps,
                                                                                            a_diffusion_coefficient)
    
    assert np.allclose(expected_diagonal, actual_diagonal)
    assert np.allclose(expected_lower_diagonal, actual_lower)
    assert np.allclose(expected_upper_diagonal, actual_upper)


def test_create_diags_for_system_matrix_robin_neumann():
    a_nb_space_points = 5
    some_discrete_steps = (0.1, 0.2)
    a_diffusion_coefficient = 3
    some_transport_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    a_transport_variable = 5
    a_nucleus_permeability = 6
    some_coefficients = (a_diffusion_coefficient, some_transport_values, a_transport_variable, a_nucleus_permeability)

    expected_diagonal = [1.4, 16.2, 16.25, 16.3, -1.0]
    expected_lower_diagonal = [-7.5, -7.5, -7.5, 0.95333]
    expected_upper_diagonal = [-1.333333, -7.75, -7.8, -7.85]

    actual_diagonal, actual_lower, actual_upper = sp.create_diags_for_system_matrix_robin_neumann(a_nb_space_points,
                                                                                                  some_discrete_steps,
                                                                                                  some_coefficients)
    assert np.allclose(expected_diagonal, actual_diagonal)
    assert np.allclose(expected_lower_diagonal, actual_lower)
    assert np.allclose(expected_upper_diagonal, actual_upper)


def test_create_reaction_array():
    some_monomer_values = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    some_dimer_values = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    some_apoe_values = np.array([3.1, 3.2, 3.3, 3.4, 3.5])
    some_complex_values = np.array([4.1, 4.2, 4.3, 4.4, 4.5])
    some_system_values = (some_monomer_values, some_dimer_values, some_apoe_values, some_complex_values)
    a_k_value = 5
    a_ka_value = 6
    a_fragmentation_for_complexes = 7
    a_fragmentation_for_dimers = 8
    some_parameters_values = (a_k_value, a_ka_value, a_fragmentation_for_complexes, a_fragmentation_for_dimers)

    expected_reaction_array = np.array([-1.1, 34.36, 32.71, 30.84, -1.5])

    actual_reaction_array = sp.create_reaction_array(some_system_values, some_parameters_values)

    assert np.allclose(expected_reaction_array, actual_reaction_array)


if __name__ == "__main__":
    test_create_sparse_matrix_from_diags()

    test_compute_transport_values_over_space()

    test_create_diags_for_system_matrix_neumann()

    test_create_diags_for_system_matrix_robin_neumann()

    test_create_reaction_array()
