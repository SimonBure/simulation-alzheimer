import numpy as np
import scipy.sparse as sparse
from matplotlib import pyplot as plt
from scipy.sparse.linalg import factorized


def spatial_simulation_one_dimension(system_parameters, initial_conditions, simulation_parameters):
    (k, ka, diffusion_parameters, transport_parameters, statin_parameters, fragmentation_parameters,
     permeability_parameters) = system_parameters
    diffusion_aox, diffusion_natural, diffusion_values_over_time = diffusion_parameters
    transport_natural, transport_radiation, transport_values_over_time = transport_parameters
    fragmentation_dimers_over_time, fragmentation_complexes_over_time = fragmentation_parameters
    statin_dose, statin_permeability_coef = statin_parameters
    alpha1, alpha2 = permeability_parameters
    m_init, d_init, a_init, c_init = initial_conditions
    duration, time_step, space_size, nb_mesh_points = simulation_parameters
    spatial_array = np.linspace(start=0, stop=space_size, num=nb_mesh_points)
    spatial_step = spatial_array[1] - spatial_array[0]

    nb_steps_simulation = int(duration / time_step)

    arrays_actual, arrays_new, arrays_display = setup_system_arrays((nb_steps_simulation, nb_mesh_points),
                                                                    initial_conditions)
    m_actual, d_actual, a_actual, c_actual = arrays_actual
    m_new, d_new, a_new, c_new = arrays_new
    m_display, d_display, a_display, c_display = arrays_display

    initial_atm_mass = compute_atm_mass(initial_conditions)
    atm_flux_nucleus = np.zeros_like(fragmentation_dimers_over_time)

    nucleus_permeability = 0
    solver_natural = create_fast_edp_solver_robin_neumann(nb_mesh_points, spatial_step, time_step, diffusion_natural,
                                                          transport_natural, nucleus_permeability)
    solver_antioxidant = create_fast_edp_solver_robin_neumann(nb_mesh_points, spatial_step, time_step, diffusion_aox,
                                                              0, nucleus_permeability)
    solver_radiation = create_fast_edp_solver_robin_neumann(nb_mesh_points, spatial_step, time_step, diffusion_natural,
                                                            transport_radiation, nucleus_permeability)
    solver_radiation_and_aox = create_fast_edp_solver_robin_neumann(nb_mesh_points, spatial_step, time_step,
                                                                    diffusion_aox, transport_radiation,
                                                                    nucleus_permeability)
    all_solvers = solver_natural, solver_antioxidant, solver_radiation, solver_radiation_and_aox

    time_simulation = 1
    array_index = 1

    while time_simulation * time_step < duration:
        fragmentation_dimers = fragmentation_dimers_over_time[time_simulation]
        fragmentation_complexes = fragmentation_complexes_over_time[time_simulation]
        diffusion_coefficient = diffusion_values_over_time[time_simulation]
        transport_coefficient = transport_values_over_time[time_simulation]

        arrays_actual = m_actual, d_actual, a_actual, c_actual
        arrays_new = m_new, d_new, a_new, c_new
        copy_new_into_actual(arrays_actual, arrays_new)

        system_values = (m_actual, d_actual, a_actual, c_actual)
        parameter_values = (k, ka, fragmentation_dimers, fragmentation_complexes)

        d_new = d_actual + time_step * (0.5 * k * m_actual**2 - fragmentation_dimers * d_actual)
        a_new = a_actual + time_step * (-ka * m_actual * a_actual + fragmentation_complexes * c_actual)
        c_new = c_actual + time_step * (ka * m_actual * a_actual - fragmentation_complexes * c_actual)

        monomers_reaction = create_monomers_reaction_array(system_values, parameter_values)

        protein_density_on_nucleus = compute_protein_density_on_nucleus((d_new, a_new, c_new))
        nucleus_permeability = compute_nucleus_permeability(permeability_parameters, protein_density_on_nucleus,
                                                            statin_parameters)
        natural_coefs = diffusion_natural, transport_natural
        actual_coefs = diffusion_coefficient, transport_coefficient
        m_new = compute_new_monomers_density(m_actual, monomers_reaction, nucleus_permeability, all_solvers,
                                             time_step, actual_coefs, natural_coefs)

        # TODO Finish the loop
        time_simulation += 1
        array_index += 1


def setup_system_arrays(simulation_parameters, initial_conditions):
    nb_steps_simulation, nb_mesh_points = simulation_parameters
    m_init, d_init, a_init, c_init = initial_conditions

    m_array = np.zeros(nb_mesh_points)
    d_array = np.zeros(nb_mesh_points)
    a_array = np.zeros(nb_mesh_points)
    c_array = np.zeros(nb_mesh_points)

    m_array_new = np.zeros(nb_mesh_points)
    d_array_new = np.zeros(nb_mesh_points)
    a_array_new = np.zeros(nb_mesh_points)
    c_array_new = np.zeros(nb_mesh_points)

    np.copyto(m_array, m_init)
    np.copyto(d_array, d_init)
    np.copyto(a_array, a_init)
    np.copyto(c_array, c_init)

    np.copyto(m_array_new, m_array)
    np.copyto(m_array_new, d_array)
    np.copyto(m_array_new, a_array)
    np.copyto(m_array_new, c_array)

    m_array_display = np.zeros([nb_steps_simulation, nb_mesh_points])
    m_array_display[0, :] = m_array
    d_array_display = np.zeros([nb_steps_simulation, nb_mesh_points])
    d_array_display[0, :] = d_array
    a_array_display = np.zeros([nb_steps_simulation, nb_mesh_points])
    a_array_display[0, :] = a_array
    c_array_display = np.zeros([nb_steps_simulation, nb_mesh_points])
    c_array_display[0, :] = c_array

    arrays_actual = m_array, d_array, a_array, c_array
    arrays_new = m_array_new, d_array_new, a_array_new, c_array_new
    arrays_display = m_array_display, d_array_display, a_array_display, c_array_display

    return arrays_actual, arrays_new, arrays_display


def compute_atm_mass(system_conditions):
    # TODO Modify with different mass for the different proteins ?
    atm_monomers, atm_dimers, _, complexes = system_conditions
    return np.sum(atm_monomers + 2 * atm_dimers + complexes)


def create_fast_edp_solver_robin_neumann(nb_space_points, space_step, time_step, diffusion_coefficient,
                                         transport_variable, nucleus_permeability):
    # TODO Regroup parameters
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


def copy_new_into_actual(actual_arrays, new_arrays):
    m_actual, d_actual, a_actual, c_actual = actual_arrays
    m_new, d_new, a_new, c_new = new_arrays
    np.copyto(m_actual, m_new)
    np.copyto(d_actual, d_new)
    np.copyto(a_actual, a_new)
    np.copyto(c_actual, c_new)


def create_monomers_reaction_array(system_values, parameters_values):
    monomers, dimers, apoe_proteins, complexes = system_values
    k, ka, fragmentation_for_complexes, fragmentation_for_dimers = parameters_values
    reaction_array = np.zeros_like(monomers)

    reaction_array[1:-1] = (-k * monomers[1:-1] * monomers[1:-1] - ka * monomers[1:-1] * apoe_proteins[1:-1]
                            + fragmentation_for_complexes * complexes[1:-1]
                            + fragmentation_for_dimers * 2 * dimers[1:-1])

    reaction_array[0] = -monomers[0]
    reaction_array[-1] = -monomers[-1]

    return reaction_array


def compute_protein_density_on_nucleus(proteins_densities):
    atm_dimers, apoe_proteins, complexes = proteins_densities
    atm_mass = 350  # kDa
    apoe_mass = 34  # kDa
    protein_mass_ratio = 34 / 350  # reference is the mass of the ATM protein
    return 2 * atm_dimers[0] + protein_mass_ratio * apoe_proteins[0] + (1 + protein_mass_ratio) * complexes[0]


def compute_nucleus_permeability(permeability_parameters, density_around_nucleus, statin_parameters):
    alpha1, alpha2 = permeability_parameters
    statin_efficiency, statin_dose = statin_parameters
    permeability = alpha1 - alpha2 * density_around_nucleus * (1 + statin_efficiency * statin_dose)

    if permeability < 0:
        permeability = 0

    return permeability


def compute_new_monomers_density(monomers_actual, monomers_reaction, nucleus_permeability, all_solvers, time_step,
                                 actual_coefs, natural_coefs, nb_space_points, space_step):
    solver_natural, solver_antioxidant, solver_radiation, solver_radiation_and_aox = all_solvers
    diffusion_coef, transport_coef = actual_coefs
    diffusion_natural, transport_natural = natural_coefs
    if nucleus_permeability == 0:
        if diffusion_coef == diffusion_natural:
            if transport_coef == transport_natural:
                monomers_new = solver_natural(monomers_actual + time_step * monomers_reaction)
            else:
                monomers_new = solver_radiation(monomers_actual + time_step * monomers_reaction)
        else:
            if transport_coef == 0:
                monomers_new = solver_antioxidant(monomers_actual + time_step * monomers_reaction)
            else:
                monomers_new = solver_radiation_and_aox(monomers_actual + time_step * monomers_reaction)
    else:
        solver_with_atm_flux = create_fast_edp_solver_robin_neumann(nb_space_points, space_step, time_step,
                                                                    diffusion_coef, transport_coef,
                                                                    nucleus_permeability)
        monomers_new = solver_with_atm_flux(monomers_actual + time_step * monomers_reaction)

    return monomers_new


def create_diffusion_coefficient(nb_time_points, time_step, natural_diffusion, aox_experiment):
    d_over_time = natural_diffusion * np.ones(nb_time_points)
    aox_time, diffusion_aox = aox_experiment

    index_start_aox = map_time_to_index(aox_time[0], time_step)
    index_end_aox = map_time_to_index(aox_time[1], time_step)
    d_over_time[index_start_aox:index_end_aox] = diffusion_aox

    return d_over_time


def create_fragmentation_coefficient(fragmentation_coefficients, nb_time_points, time_step, experimental_conditions,
                                     proba_stress_peak):
    (fragmentation_irradiation, fragmentation_aox, fragmentation_stress_peak,
     fragmentation_natural) = fragmentation_coefficients

    tau_over_time = fragmentation_natural * np.ones(nb_time_points)

    indexes_stress_peaks = get_indexes_stress_peaks(nb_time_points, proba_stress_peak)
    # The stress is either natural or a peak of stress
    tau_over_time[indexes_stress_peaks] = fragmentation_stress_peak

    irradiation, antioxidant, _ = experimental_conditions
    irr_time, transport_irr = irradiation
    aox_time, transport_aox = antioxidant

    index_start_irr = map_time_to_index(irr_time[0], time_step)
    index_end_irr = map_time_to_index(irr_time[1], time_step)
    # Stresses of irradiation or antioxidants add to natural or peak stresses
    tau_over_time[index_start_irr:index_end_irr] += fragmentation_irradiation

    index_start_aox = map_time_to_index(aox_time[0], time_step)
    index_end_aox = map_time_to_index(aox_time[1], time_step)
    tau_over_time[index_start_aox:index_end_aox] += fragmentation_aox

    return tau_over_time


def get_indexes_stress_peaks(nb_time_points, proba_stress_peak):
    is_peak_on_time_index = np.random.random(nb_time_points) < proba_stress_peak
    indexes_peaks = np.nonzero(is_peak_on_time_index)
    return indexes_peaks


def create_transport_coefficient(spatial_space, natural_transport, transport_constant, experimental_conditions,
                                 time_step, nb_time_points):
    v_over_space = np.exp(-transport_constant * spatial_space)
    irradiation, antioxidant, _ = experimental_conditions
    irr_time, transport_irr = irradiation
    aox_time, transport_aox = antioxidant

    v_over_time = natural_transport * np.ones(nb_time_points)

    index_start_irr = map_time_to_index(irr_time[0], time_step)
    index_end_irr = map_time_to_index(irr_time[1], time_step)
    v_over_time[index_start_irr:index_end_irr] = transport_irr

    index_start_aox = map_time_to_index(aox_time[0], time_step)
    index_end_aox = map_time_to_index(aox_time[1], time_step)
    v_over_time[index_start_aox:index_end_aox] = transport_aox

    return v_over_time * v_over_space


def map_time_to_index(time_step, time):
    return int(time / time_step)


if __name__ == "__main__":
    # Simulation parameters
    a_space_size = 10
    a_nb_mesh_points = 1000
    a_space_array = np.linspace(0, a_space_size, a_nb_mesh_points)
    a_space_step = a_space_array[1] - a_space_array[0]
    a_time_step = 0.1
    a_duration = 35
    flash = 1
    Bool_flash = True

    # System parameters
    a_diffusion_natural = 1
    DO = 2
    a_k = 1
    a_ka = 1
    fragmentation_radiation = 9
    fragmentation_antioxidant = 4
    diff_tau = 1.1
    a_transport_radiation = 0.1
    a_transport_stress = 0.05
    Cond_Stress = True
    stress_intensity_peak = 2.5
    stress_continuous = 0.2
    stress_peak_probability = 0.95
    a_alpha1 = 1
    a_alpha2 = 0.45
    some_permeability_coefs = (a_alpha1, a_alpha2)
    statin_efficiency = 1.2

    # TODO What is that ?
    rCI0 = 1
    rD = 3

    # Initial conditions
    initial_dimers_density = 1
    p10f = np.zeros([a_nb_mesh_points])
    p2a0f = np.zeros([a_nb_mesh_points])
    p20f = initial_dimers_density * np.ones([a_nb_mesh_points])
    p1a0f = np.zeros([a_nb_mesh_points])
    p1a0f = np.exp(-rCI0 * a_space_array)
