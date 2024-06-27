import compartmental as cp
import numpy as np


def test_setup_initial_compartments_values():
    a_crown_configuration = 'formed'
    assert cp.setup_initial_compartments_values(a_crown_configuration) == (300, 0, 0, 0, 0, 200, 300)

    a_crown_configuration = 'not formed'
    assert cp.setup_initial_compartments_values(a_crown_configuration) == (300, 0, 0, 0, 200, 0, 0)

    a_initial_condition = (1, 2, 3, 4, 5, 6, 7)
    assert cp.setup_initial_compartments_values(a_initial_condition) == (1, 2, 3, 4, 5, 6, 7)


def test_setup_display_arrays():
    a_size_array = 10
    initial_condition = (1, 2, 3, 4, 5, 6, 7)
    expected_output = [np.full(a_size_array, 1), np.full(a_size_array, 2),
                       np.full(a_size_array, 3), np.full(a_size_array, 4),
                       np.full(a_size_array, 5), np.full(a_size_array, 6),
                       np.full(a_size_array, 7)]
    actual_output = cp.setup_display_arrays(initial_condition, a_size_array)

    for i, tab in enumerate(actual_output):
        for j, value in enumerate(tab):
            assert value == expected_output[i][j]


def test_fill_display_arrays():
    some_array_to_fill = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    some_compartments_values = (10, 11, 12)
    index_to_fill = 2
    expected_output = [np.array([1, 2, 10]), np.array([4, 5, 11]), np.array([7, 8, 12])]
    actual_output = cp.fill_display_arrays(some_array_to_fill, index_to_fill, some_compartments_values)

    for i, tab in enumerate(actual_output):
        for j, value in enumerate(tab):
            assert value == expected_output[i][j]


def test_dose_antioxidant():
    a_antioxidant_using = 'variable'
    some_antioxidant_parameters = ((1, 2, 3), 4)
    expected_output = 1 * np.exp(-(4 - 2) ** 2 / (2 * 3 ** 2))
    assert cp.dose_antioxidant(a_antioxidant_using, some_antioxidant_parameters) == expected_output

    a_antioxidant_using = 'constant'
    assert cp.dose_antioxidant(a_antioxidant_using, some_antioxidant_parameters) == 1

    a_antioxidant_using = 'none'
    assert cp.dose_antioxidant(a_antioxidant_using, some_antioxidant_parameters) == 0


def test_dose_statin():
    a_statin_using = 'variable'
    some_statin_parameters = ((1, 2, 3), 4)
    expected_output = 1 * np.exp(-(4 - 2) ** 2 / (2 * 3 ** 2))
    assert cp.dose_statin(a_statin_using, some_statin_parameters) == expected_output

    a_statin_using = 'constant'
    assert cp.dose_statin(a_statin_using, some_statin_parameters) == 1

    a_statin_using = 'none'
    assert cp.dose_statin(a_statin_using, some_statin_parameters) == 0


def test_dose_stress():
    some_stress_conditions = (False, False)
    some_stress_parameters = ((1, 2), (3, 4, 5), 6, 7)
    assert cp.dose_stress(some_stress_conditions, some_stress_parameters) == 0

    some_stress_conditions = (True, False)
    assert cp.dose_stress(some_stress_conditions, some_stress_parameters) == 1 / (1 + 2 * 6)

    some_stress_conditions = (True, True)
    expected_output = 1 / (1 + 2 * 6) + 3 * np.exp(-(7 - 4) ** 2 / (2 * 5 ** 2))
    assert cp.dose_stress(some_stress_conditions, some_stress_parameters) == expected_output


def test_update_system_rates():
    some_compartments_values = (1, 2, 3, 4, 5, 6, 7)

    some_migration_pc_coefficients = (8, 9, 10, 11)
    some_migration_nucleus_coefficients = (12, 13, 14, 15, 16)
    some_complex_formation_coefficients = (17, 18, 19, 20)
    a_a1, a_e1, a_a4, a_e4, a_e6 = (21, 22, 23, 24, 25)
    some_stress_coefficients = (26, 27)
    some_irradiation_coefficients = (28, 29, 30)
    some_coefficients = (some_migration_pc_coefficients, some_migration_nucleus_coefficients,
                         some_complex_formation_coefficients, (a_a1, a_e1, a_a4, a_e4, a_e6),
                         some_stress_coefficients, some_irradiation_coefficients)
    some_drugs_doses = (31, 32)
    some_stress_conditions = (True, True)
    a_time = 33
    expected_k1 = 21 / (1 + 22 * 31)
    expected_k2 = ((9 * 8 ** 10) / (8 ** 10 + 7 ** 10)) / (1 + 11 * 31)
    expected_k3 = ((13 * 12 ** 14) / (12 ** 14 + 7 ** 14)) * (1 + 15 * 31) * (1 + 16 * 32)
    expected_k4 = 23 / (1 + 24 * 31)
    expected_k5 = ((17 * 6 ** 19) / (18 ** 19 + 6 ** 19)) / (1 + 20 * 31)
    expected_k6 = 25 * 31
    expected_g = (26 / (1 + 27 * 31)) + 28 * np.exp(-(33 - 29)**2 / (2 * 30 ** 2))

    (actual_k1, actual_k2, actual_k3, actual_k4,
     actual_k5, actual_k6, actual_g) = cp.update_system_rates(some_compartments_values, some_coefficients,
                                                              some_drugs_doses, some_stress_conditions, a_time)
    assert expected_k1 == actual_k1
    assert expected_k2 == actual_k2
    assert expected_k3 == actual_k3
    assert expected_k4 == actual_k4
    assert expected_k5 == actual_k5
    assert expected_k6 == actual_k6
    assert expected_g == actual_g


def test_compute_discriminant():
    some_parameters = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    expected_value = (((6 - 2 * 9) * (2 + 10))**2 + 8 * 4 * 2 * 10 * 1 * ((6 - 9) / 5)**2)
    actual_value = cp.compute_discriminant(some_parameters)
    assert expected_value == actual_value


def test_compute_fixed_point_without_stress():
    some_parameters = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    expected_value = {'Dc': 1 / 2, 'Mc': 0, 'Ma': 0, 'Mn': 0, 'A': 'A*', 'Ca': 'Ca*', 'Da': 'Da*'}
    actual_value = cp.compute_fixed_point_without_stress(some_parameters)
    assert expected_value == actual_value


def test_compute_fixed_point_with_stress():
    some_parameters = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    some_initial_a0_ca0 = (11, 12)
    a_disc = cp.compute_discriminant(some_parameters)
    expected_ma_eq_plus = 5 ** 2 * (2 * 9 - 6 + np.sqrt(a_disc)) / (2 * 4 * 2 * (6 - 9) ** 2)
    expected_mc_eq_plus = ((6 - 9) / 5) * expected_ma_eq_plus
    expected_mn_eq_plus = (6 / 3) * expected_ma_eq_plus
    expected_da_eq_plus = (8 / (2 * 10)) * expected_ma_eq_plus ** 2
    expected_dc_eq_plus = (1 + (4 / 2) * expected_mc_eq_plus ** 2) / (2 + 10)
    expected_a_eq_plus = 10 * (11 + 12) / (10 + 7 * expected_ma_eq_plus)
    expected_ca_eq_plus = 11 + 12 - expected_a_eq_plus
    expected_fixed_point_plus = {'Dc': expected_dc_eq_plus, 'Mc': expected_mc_eq_plus, 'Ma': expected_ma_eq_plus,
                                 'Mn': expected_mn_eq_plus, 'A': expected_a_eq_plus, 'Ca': expected_ca_eq_plus,
                                 'Da': expected_da_eq_plus}

    expected_ma_eq_neg = 5 ** 2 * (2 * 9 - 6 - np.sqrt(a_disc)) / (2 * 4 * 2 * (6 - 9) ** 2)
    expected_mc_eq_neg = ((6 - 9) / 5) * expected_ma_eq_neg
    expected_mn_eq_neg = (6 / 3) * expected_ma_eq_neg
    expected_da_eq_neg = (8 / (2 * 10)) * expected_ma_eq_neg ** 2
    expected_dc_eq_neg = (1 + (4 / 2) * expected_mc_eq_neg ** 2) / (2 + 10)
    expected_a_eq_neg = 10 * (11 + 12) / (10 + 7 * expected_ma_eq_neg)
    expected_ca_eq_neg = 11 + 12 - expected_a_eq_neg
    expected_fixed_point_neg = {'Dc': expected_dc_eq_neg, 'Mc': expected_mc_eq_neg, 'Ma': expected_ma_eq_neg,
                                'Mn': expected_mn_eq_neg, 'A': expected_a_eq_neg, 'Ca': expected_ca_eq_neg,
                                'Da': expected_da_eq_neg}

    actual_fixed_point_plus, actual_fixed_point_neg = cp.compute_fixed_points_with_stress(some_parameters,
                                                                                          some_initial_a0_ca0)
    assert expected_fixed_point_plus == actual_fixed_point_plus
    assert expected_fixed_point_neg == actual_fixed_point_neg


if __name__ == '__main__':
    test_setup_initial_compartments_values()

    test_setup_display_arrays()

    test_fill_display_arrays()

    test_dose_antioxidant()

    test_dose_statin()

    test_dose_stress()

    test_update_system_rates()

    test_compute_discriminant()

    test_compute_fixed_point_without_stress()

    test_compute_fixed_point_with_stress()
