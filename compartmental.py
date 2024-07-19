import numpy as np
from matplotlib import pyplot as plt


def compartmental_simulation(duration: float, time_step: float, initial_crown_configuration,
                             experimental_conditions: tuple) -> tuple:
    """
    Simulate the evolution of the Alzheimer compartmental model for the given duration.
    :param duration: float giving the time when the simulation must stop, given in hours.
    :param time_step: float representing the time step between 2 computations, given in hours. Default to 1 minute.
    :param initial_crown_configuration: str indicating which initial condition to use. Can be either 'formed', 'f' if
        you want the perinuclear crown to be formed initially, 'not formed', 'nf', 'no', 'n' if not.
        You can also give a tuple containing the initial conditions if you want to use customized initial conditions.
    :param experimental_conditions: tuple containing the indications on the experimental conditions.
        Elements are the following:
        1. irradiation: boolean indicating whether to use the irradiation or not. Default to False.
        2. antioxidant_use: string indicating the way antioxidant are used. Could be 'none', 'no', 'n' if no
            antioxidant are used. Can also be 'constant', 'const', 'cst', 'c' if the dose is constant or 'variable',
            'var', 'v' if the dose vary with the time of the simulation. Default to 'none'.
        3. statin_use: string indicating the way statin are used. Could be 'none', 'no', 'n' if no statins are used.
            Can also be 'constant', 'const', 'cst', 'c' if the dose is constant or 'variable', 'var', 'v' if the dose
            vary with the time of the simulation. Default to 'none'.
        4. stress: boolean indicating whether the cell experiments stress or not. Default to True.
    :return: first a dictionary containing the data arrays of all the 7 compartments and then a dict for the fixed point
    """

    is_irradiation, antioxidant_using, statin_using, is_stress = experimental_conditions

    # Coefficients definition
    lam = 15  # constant source of ATM dimers in cytoplasm
    d0 = 0.05  # rate of ATM dimers degradation
    d1 = 0.3  # rate of ATM monomers degradation in the nucleus
    a1 = 0.01  # rate of ATM monomers re-dimerisation
    e0 = e1 = e2 = e4 = e5 = 20  # inhibiting impact of the antioxidant
    e3 = e6 = 0.5  # promoting impact of the antioxidant
    f3 = 1  # impact of statin on nucleus permeability

    k1, k2, k3, k4, k5, k6, g = 0, 0, 0, 0, 0, 0, 0

    migration_pc_coefficients = (400, 0.4, 15, e2)  # coefficients for the hill function k2
    migration_nucleus_coefficients = (80, 0.5, 5, e3, f3)  # coefficients for the hill function k3

    a4 = 0.05  # rate of ATM-ApoE complexes formation
    complex_formation_coefficients = (0.4, 150, 15, e5)

    cs = 0.002  # rate of protein dissociation during constant stress
    stress_coefficients = (cs, e0)

    cox = 1.5  # dose of antioxidant
    antioxidant_coefficients = (cox, 9, 3)  # parameters for the gaussian representation of the antioxidant effect

    sta = 5  # dose of statin
    statin_coefficients = (sta, 9, 4)

    irradiation_coefficients = (0.8, 9, 2)  # parameters for the gaussian representation of the irradiation stress

    initial_compartments_values = compartments_values = dc, mc, ma, mn, a, ca, da = setup_initial_compartments_values(
        initial_crown_configuration)

    nb_loops_simulation = int(duration / time_step)

    # We need an extra slot because initial conditions are in the first slot of compartments_display_arrays
    compartments_display_arrays = setup_display_arrays(compartments_values, nb_loops_simulation)

    time_simulation = 0

    for display_array_index in range(1, nb_loops_simulation):
        # TODO Class for parameters & the system of 7 compartments
        parameters_antioxidant_dosage = (antioxidant_coefficients, time_simulation)
        antioxidant_dose = dose_antioxidant(antioxidant_using, parameters_antioxidant_dosage)

        parameters_statin_dosage = (statin_coefficients, time_simulation)
        statin_dose = dose_statin(statin_using, parameters_statin_dosage)

        drugs_doses = (antioxidant_dose, statin_dose)

        stress_conditions = (is_stress, is_irradiation)
        coefficients_update_rates = (migration_pc_coefficients, migration_nucleus_coefficients,
                                     complex_formation_coefficients, (a1, e1, a4, e4, e6), stress_coefficients,
                                     irradiation_coefficients)
        k1, k2, k3, k4, k5, k6, g = update_system_rates(compartments_values, coefficients_update_rates, drugs_doses,
                                                        stress_conditions, time_simulation)

        # Update rule using Euler explicit numerical scheme
        dc += time_step * (lam - d0 * dc + (0.5 * k1 * mc ** 2) - g * dc)
        mc += time_step * ((-k1 * mc ** 2) - k2 * mc + k6 * ma + 2 * g * dc)
        ma += time_step * (k2 * mc - k3 * ma - k4 * a * ma - (k5 * ma ** 2) - k6 * ma + 2 * g * da + g * ca)
        mn += time_step * (k3 * ma - d1 * mn)
        a += time_step * (-k4 * ma * a + g * ca)
        ca += time_step * (k4 * ma * a - g * ca)
        da += time_step * ((0.5 * k5 * ma ** 2) - g * da)

        compartments_values = dc, mc, ma, mn, a, ca, da

        # TODO Refactor into a class of display arrays
        compartments_display_arrays = fill_display_arrays(compartments_display_arrays, display_array_index,
                                                          compartments_values)

        time_simulation += time_step
        display_array_index += 1

    time_array = np.linspace(0, duration, num=nb_loops_simulation)

    # To use for fixed point computations
    all_parameters_system = (lam, d0, d1, k1, k2, k3, k4, k5, k6, g)

    fixed_point_no_stress = compute_fixed_point_without_stress(all_parameters_system)

    if is_stress:
        initial_conditions_for_fixed_point = (initial_compartments_values[4], initial_compartments_values[5])
        fixed_points_stress = compute_fixed_points_with_stress(all_parameters_system,
                                                               initial_conditions_for_fixed_point)
    else:
        fixed_points_stress = None

    get_existence_conditions(all_parameters_system)

    return time_array, compartments_display_arrays, fixed_points_stress, fixed_point_no_stress


def setup_initial_compartments_values(initial_crown_configuration) -> tuple:
    # Allow abbreviations for initial_crown_configuration
    crown_formed = ('formed', 'f')
    crown_not_formed = ('not formed', 'nf', 'no', 'n')

    is_initial_conditions_in_correct_format(initial_crown_configuration)

    if type(initial_crown_configuration) is str:
        dc_initial = 300
        mc_initial = 0
        ma_initial = 0
        mn_initial = 0
        a_initial = 200 if initial_crown_configuration in crown_not_formed else 0
        ca_initial = 200 if initial_crown_configuration in crown_formed else 0
        da_initial = 300 if initial_crown_configuration in crown_formed else 0

    elif type(initial_crown_configuration) is tuple:
        dc_initial, mc_initial, ma_initial, mn_initial, a_initial, ca_initial, da_initial = initial_crown_configuration

    else:
        dc_initial, mc_initial, ma_initial, mn_initial, a_initial, ca_initial, da_initial = 0, 0, 0, 0, 0, 0, 0

    return dc_initial, mc_initial, ma_initial, mn_initial, a_initial, ca_initial, da_initial


def is_initial_conditions_in_correct_format(initial_crown_configuration):
    crown_formed = ('formed', 'f')
    crown_not_formed = ('not formed', 'nf', 'no', 'n')

    if type(initial_crown_configuration) is str:
        if initial_crown_configuration not in crown_formed and initial_crown_configuration not in crown_not_formed:
            raise ValueError(f"Parameter [initial_crown_configuration] must be one of the following:\n"
                             f"{crown_formed} or {crown_not_formed}.\n"
                             f"Input value: {initial_crown_configuration}")
    elif type(initial_crown_configuration) is tuple:
        if len(initial_crown_configuration) != 7:
            raise ValueError(f"Parameter [initial_crown_configuration] must contain initial values for all seven"
                             f"compartments")
    else:
        raise TypeError(f"Parameter [initial_crown_configuration] must either be of type str or tuple.\n"
                        f"Input type: {type(initial_crown_configuration)}")


def setup_display_arrays(initial_compartments_values, size_arrays) -> []:
    compartment_display_arrays = [0] * 7
    for (index, compartment) in enumerate(initial_compartments_values):
        compartment_display_arrays[index] = np.full(size_arrays, compartment)

    return compartment_display_arrays


def fill_display_arrays(compartment_display_arrays, index, compartments_values):
    for i, compartment in enumerate(compartment_display_arrays):
        compartment[index] = compartments_values[i]

    return compartment_display_arrays


def dose_antioxidant(antioxidant_using, antioxidant_dosage_parameters) -> float:
    coefficients_antioxidant = antioxidant_dosage_parameters[0]
    time_simulation = antioxidant_dosage_parameters[1]

    # Allow abbreviation for antioxidant_using
    no_antioxidant = ('none', 'no', '0', 'n')
    constant_antioxidant = ('constant', 'const', 'cst', 'c')
    variable_antioxidant = ('variable', 'var', 'v')

    if antioxidant_using in no_antioxidant:
        antioxidant_dosage = 0
    elif antioxidant_using in constant_antioxidant:
        antioxidant_dosage = coefficients_antioxidant[0]
    elif antioxidant_using in variable_antioxidant:
        antioxidant_dosage = compute_antioxidant_time_dose(time_simulation, coefficients_antioxidant)
    else:
        raise ValueError("Please select a valid use for the antioxidant")

    return antioxidant_dosage


def compute_antioxidant_time_dose(time: float, coefficients: tuple) -> float:
    """
    Calculates the value of the antioxidant effects using a Gaussian function.
    :param time: float giving the actual time of the simulation.
    :param coefficients: tuple of 3 elements containing the coefficient of the Gaussian function. Must be in order
        c_aox, m_aox, sigma_aox.
    :return: float representing the value of the antioxidant effects.
    """
    c_aox = coefficients[0]  # antioxidant dose
    m_aox = coefficients[1]  # time when the antioxidant are given
    sigma_aox = coefficients[2]  # rapidity of the antioxidant effects

    return c_aox * np.exp(-(time - m_aox) ** 2 / (2 * sigma_aox ** 2))


def dose_statin(statin_using, statin_dosage_parameters) -> float:
    coefficients_statin = statin_dosage_parameters[0]
    time_simulation = statin_dosage_parameters[1]

    # Allow abbreviation for statin_using
    no_statin = ('none', 'no', '0', 'n')
    constant_statin = ('constant', 'const', 'cst', 'c')
    variable_statin = ('variable', 'var', 'v')

    if statin_using in no_statin:
        statin_dosage = 0
    elif statin_using in constant_statin:
        statin_dosage = coefficients_statin[0]
    elif statin_using in variable_statin:
        statin_dosage = compute_statin_time_dose(time_simulation, coefficients_statin)
    else:
        raise ValueError("Please select a valid use for the statin")

    return statin_dosage


def compute_statin_time_dose(time: float, coefficients: tuple) -> float:
    """
    Calculates the value of the statin effects using a Gaussian function.
    :param time: float giving the actual time of the simulation.
    :param coefficients: tuple of 3 elements containing the coefficient of the Gaussian function. Must be in order
            s_ta, m_ta, sigma_ta.
    :return: float representing the value of the statin effects.
    """
    s_ta = coefficients[0]  # statin dose
    m_ta = coefficients[1]  # time when the statin are taken
    sigma_ta = coefficients[2]  # rapidity of the statin effects or biocompatibility

    return s_ta * np.exp(-(time - m_ta) ** 2 / (2 * sigma_ta ** 2))


def update_system_rates(compartments_values: tuple, coefficients: tuple, drugs_doses: tuple, stress_conditions: tuple,
                        time_simulation: float) -> tuple:
    dc, mc, ma, mn, a, ca, da = compartments_values

    antioxidant_dose = drugs_doses[0]

    migration_pc_coefficients = coefficients[0]
    migration_nucleus_coefficients = coefficients[1]
    complex_formation_coefficients = coefficients[2]
    a1, e1, a4, e4, e6 = coefficients[3]
    stress_coefficients = coefficients[4]
    irradiation_coefficients = coefficients[5]

    k1 = a1 / (1 + e1 * antioxidant_dose)  # total rate of dimers formation
    # Migration rate of the ATM monomers from cytoplasm into the PC
    k2 = compute_migration_rate_to_pc(da, migration_pc_coefficients, drugs_doses)
    # Migration rate of the ATM monomers from PC into the nucleus
    k3 = compute_migration_rate_to_nucleus(da, migration_nucleus_coefficients, drugs_doses)
    k4 = a4 / (1 + e4 * antioxidant_dose)
    # Dimers formation  rate of the ATM monomers inside the PC
    k5 = compute_dimers_formation_rate(ca, complex_formation_coefficients, antioxidant_dose)
    k6 = e6 * antioxidant_dose  # dispersion caused by the antioxidant

    parameters_stress_dosage = (stress_coefficients, irradiation_coefficients, antioxidant_dose, time_simulation)
    g = dose_stress(stress_conditions, parameters_stress_dosage)

    return k1, k2, k3, k4, k5, k6, g


def compute_migration_rate_to_pc(monomers_concentration: float, migration_coefficients: tuple,
                                 drugs_doses: tuple) -> float:
    """
    Function to calculate the rate of migration of monomers between cytoplasm and perinuclear crown (PC) or PC and
    nucleus.
    :param monomers_concentration: float and variable of the Hill function. Here it is the concentration of ATM dimers
        in the PC.
    :param migration_coefficients: tuple of 3 elements containing the coefficient of the Hill function.
        Must be in order a, b, n, e.
    :param drugs_doses: tuple containing the doses of antioxidant and of statin taken. Default to (0, 0).
    :return: float representing the rate of migration, in h⁻¹.
    """
    a = migration_coefficients[0]
    b = migration_coefficients[1]
    n = migration_coefficients[2]
    e = migration_coefficients[3]  # antioxidant effect

    antioxidant_dose = drugs_doses[0]

    return ((b * a ** n) / (a ** n + monomers_concentration ** n)) / (1 + e * antioxidant_dose)


def compute_migration_rate_to_nucleus(monomers_concentration, migration_coefficients, drugs_doses):
    a = migration_coefficients[0]
    b = migration_coefficients[1]
    n = migration_coefficients[2]
    e = migration_coefficients[3]  # antioxidant effect
    f = migration_coefficients[4]  # statin effect

    antioxidant_dose = drugs_doses[0]
    statin_dose = drugs_doses[1]

    return ((b * a ** n) / (a ** n + monomers_concentration ** n)) * (1 + e * antioxidant_dose) * (1 + f * statin_dose)


def compute_dimers_formation_rate(concentration: float, coefficients: tuple, antioxidant_dose: float = 0) -> float:
    """
    Hill function to calculate the rate of dimers formation in the PC.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM-ApoE complexes in
        the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n.
    :param antioxidant_dose: float indicating the quantity of antioxidant taken.
    :return: float representing the rate of dimers formation, in h⁻¹.
    """
    # Normal rate of dimers formation
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]

    # Antioxidant effect
    e = coefficients[3]
    dimers_formation_rate = ((a * concentration ** n) / (b ** n + concentration ** n)) / (1 + e * antioxidant_dose)

    return dimers_formation_rate


def dose_stress(stress_conditions, stress_dosage_parameters) -> float:
    is_stress = stress_conditions[0]
    is_irradiation = stress_conditions[1]
    cs = stress_dosage_parameters[0][0]
    e0 = stress_dosage_parameters[0][1]
    irradiation_coefficients = stress_dosage_parameters[1]
    antioxidant_dose = stress_dosage_parameters[2]
    time_simulation = stress_dosage_parameters[3]

    g = 0
    if is_stress:
        g = cs / (1 + e0 * antioxidant_dose)
        if is_irradiation:
            g += compute_irradiation_time_stress(time_simulation, irradiation_coefficients)

    return g


def compute_irradiation_time_stress(time: float, coefficients: tuple) -> float:
    """
    Calculates the value of the irradiation stress using a gaussian function.
    :param time: float giving the actual time of the simulation.
    :param coefficients: tuple of 3 elements containing the coefficient of the Gaussian function. Must be in order a_s,
        m_s, sigma_s.
    :return: float representing the value of the irradiation induced stress.
    """
    a_s = coefficients[0]  # intensity of the irradiation
    m_s = coefficients[1]  # time when the irradiation occurs
    sigma_s = coefficients[2]  # rapidity of the irradiation effects

    return a_s * np.exp(-(time - m_s) ** 2 / (2 * sigma_s ** 2))


# TO-DO Refactor display system into a class ?
def plot_compartment(simulation_results: tuple, download: bool = False):
    """
    Function of higher level to specify which compartment to plot, a specific title and whether to download the plot or
    not.
    :param simulation_results: dict containing the results of the simulation.
    :param download: boolean indicating whether to download the plot or not.
    """
    fig, (_, __) = plt.subplots(1, 2, figsize=(12, 5))

    _ = plot_cyto_nucleus(simulation_results, _)
    __ = plot_perinuclear_crown(simulation_results, __)

    plt.subplots_adjust(wspace=0.4)  # setting horizontal space between the two plots

    if download is True:
        fig.savefig(f'FIGURES/plot.png')

    plt.show()


# TO-DO Update this function to be used with the new output of
def plot_cyto_nucleus(simulation_results: tuple, ax):
    """
    Function to plot the trajectories of the simulation in the cytoplasm and the nucleus.
    :param simulation_results: dict containing the results of the simulation.
    :param ax: matplotlib.figure.Axes object passed to the function to add the trajectories on it for further display.
    :return: matplotlib.figure.Axes object with trajectories of the simulation in the cytoplasm and the nucleus added.
    """
    time_array, compartments_display_arrays = simulation_results
    # Data plotting
    ax.plot(time_array, compartments_display_arrays[0], label='$D_C$', color='blue')
    ax.plot(time_array, compartments_display_arrays[1], label='$M_C$', color='green')
    ax.plot(time_array, compartments_display_arrays[3], label='$M_N$', color='red')

    # Titles & labels
    ax.set_title("Populations evolution in the cytoplasm & nucleus")
    ax.set_xlabel('Time ($h$)', fontsize=12)
    ax.set_ylabel('Populations', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    return ax


def plot_perinuclear_crown(simulation_results, ax):
    """
    Function to plot the trajectories of the simulation in the perinuclear crown.
    :param simulation_results: dict containing the results of the simulation.
    :param ax: matplotlib.figure.Axes object passed to the function to add the trajectories on it for further display.
    :return: matplotlib.figure.Axes object with trajectories of the simulation in the perinuclear crown added.
    """
    time_array, compartments_display_arrays = simulation_results
    # Data plotting
    ax.plot(time_array, compartments_display_arrays[4], label='$A$', color='magenta')
    ax.plot(time_array, compartments_display_arrays[5], label='$C_A$', color='darkorange')
    ax.plot(time_array, compartments_display_arrays[6], label='$D_A$', color='crimson')
    ax.plot(time_array, compartments_display_arrays[2], label='$M_A$', color='dodgerblue')

    # Titles & labels
    ax.set_title("Populations evolution in the perinuclear crown")
    ax.set_xlabel('Time ($h$)', fontsize=12)
    ax.set_ylabel('Populations', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    return ax


def compute_fixed_points_with_stress(system_parameters: tuple, initial_a0_ca0: tuple) -> tuple[dict, dict]:
    """
    Function to compute the theoretical values of the equilibrium points. I
    :param system_parameters: dict containing the parameters needed to compute the equilibrium.
    :param initial_a0_ca0: tuple of initial values needed to compute the equilibrium.
    :return: If stress is True then return two dict, each containing a value for the equilibrium. If stress is False
        then return only one dict of the simple equilibrium.
    """
    lam, d0, d1, k1, k2, k3, k4, k5, k6, g = system_parameters

    a0, ca0 = initial_a0_ca0

    disc = compute_discriminant(system_parameters)

    ma_eq_plus = k2**2 * (2 * k6 - k3 + np.sqrt(disc)) / (2 * k1 * d0 * (k3 - k6)**2)
    mc_eq_plus = ((k3 - k6) / k2) * ma_eq_plus
    mn_eq_plus = (k3 / d1) * ma_eq_plus
    da_eq_plus = (k5 / (2 * g)) * ma_eq_plus**2
    dc_eq_plus = (lam + (k1 / 2) * mc_eq_plus**2) / (d0 + g)
    a_eq_plus = g * (a0 + ca0) / (g + k4 * ma_eq_plus)
    ca_eq_plus = a0 + ca0 - a_eq_plus

    ma_eq_neg = k2**2 * (2 * k6 - k3 - np.sqrt(disc)) / (2 * k1 * d0 * (k3 - k6)**2)
    mc_eq_neg = ((k3 - k6) / k2) * ma_eq_neg
    mn_eq_neg = (k3 / d1) * ma_eq_neg
    da_eq_neg = (k5 / (2 * g)) * ma_eq_neg**2
    dc_eq_neg = (lam + (k1 / 2) * mc_eq_neg ** 2) / (d0 + g)
    a_eq_neg = g * (a0 + ca0) / (g + k4 * ma_eq_neg)
    ca_eq_neg = a0 + ca0 - a_eq_neg

    fixed_point_plus = {'Dc': dc_eq_plus, 'Mc': mc_eq_plus, 'Ma': ma_eq_plus, 'Mn': mn_eq_plus, 'A': a_eq_plus,
                        'Ca': ca_eq_plus, 'Da': da_eq_plus}
    fixed_point_neg = {'Dc': dc_eq_neg, 'Mc': mc_eq_neg, 'Ma': ma_eq_neg, 'Mn': mn_eq_neg, 'A': a_eq_neg,
                       'Ca': ca_eq_neg, 'Da': da_eq_neg}

    fixed_points = fixed_point_plus, fixed_point_neg

    return fixed_points


def compute_fixed_point_without_stress(system_parameters: tuple) -> dict:
    lam, d0, d1, k1, k2, k3, k4, k5, k6, g = system_parameters
    return {'Dc': lam / d0, 'Mc': 0, 'Ma': 0, 'Mn': 0, 'A': 'A*', 'Ca': 'Ca*', 'Da': 'Da*'}


def get_existence_conditions(system_parameters: tuple):
    """
    Function to write the theoretical conditions on the parameters that are needed for the equilibria to exist.
    :param system_parameters: dict containing the parameters needed to verify the conditions.
    """
    lam, d0, d1, k1, k2, k3, k4, k5, k6, g = system_parameters

    disc = compute_discriminant(system_parameters)

    print(f"Discriminant's value:  {disc}")
    print(f"Parameters' values: λ = {lam}\td0 = {d0}\td1 = {d1}\tk1 = {k1}\tk2 = {k2}")
    print(f"k3 = {k3}\tk4 = {k4}\tk5 = {k5}\tk6 = {k6}\tg = {g}")  # not enough space on above line
    print(f"sqrt(Δ) = {np.sqrt(disc)}")
    print("Ma+ exists if 2k6 + sqrt(Δ) > k3")
    print(f"Condition fulfilled ? {2 * k6 + np.sqrt(disc) > k3}")
    print("Ma- exists if 2k6 > sqrt(Δ) + k3")
    print(f"Condition fulfilled ? {2 * k6 > np.sqrt(disc) + k3}")


def compute_discriminant(system_parameters: tuple) -> float:
    """
    Function to compute the value of the discriminant needed to get the value of the equilibrium with stress.
    :param system_parameters: dict containing the parameters needed to compute the discriminant.
    :return: the discriminant value as a float value.
    """
    lam, d0, d1, k1, k2, k3, k4, k5, k6, g = system_parameters

    discriminant = (((k3 - 2 * k6) * (d0 + g))**2 +
                    8 * k1 * d0 * g * lam *
                    ((k3 - k6) / k2)**2)
    return discriminant


if __name__ == "__main__":
    # k3_parameters = (80, 0.5, 5, 20, 0)
    # da_range = np.linspace(0, 1000, 1000)
    # some_drugs_doses = (1, 0)
    # k3_values = compute_migration_rate_to_nucleus(da_range, k3_parameters, some_drugs_doses)
    # plt.plot(da_range, k3_values)

    is_irradiated = True
    # is_irradiated = False
    antioxidant_dose_simulation = 'no'
    # antioxidant_dose_simulation = 'cst'
    # antioxidant_dose_simulation = 'var'
    statin_dose_simulation = 'no'
    # statin_dose_simulation = 'cst'
    # statin_dose_simulation = 'var'
    # is_stress = True
    is_stress_simulation = True
    some_experimental_conditions = (is_irradiated, antioxidant_dose_simulation, statin_dose_simulation,
                                    is_stress_simulation)
    # np.random.random
    # initial_conditions = (150, 0, 0, 0, 0, 0, 0)
    initial_conditions = 'formed'
    # initial_conditions = 'not formed'

    duration = 24  # hours

    a_time_array, some_compartments_results, eq_stress, eq_no_stress = compartmental_simulation(duration, 1 / 60,
                                                                                                initial_conditions,
                                                                                                some_experimental_conditions)
    print(f"Example of the simulation results:\n{some_compartments_results}")
    print(f"Equilibria when stress is considered:\n{eq_stress}")
    print(f"Equilibrium without stress:\n{eq_no_stress}")
    # print(f"Da: {test_simulation['Da']}")
    # print(f"Ma: {test_simulation['Ma']}")
    # print(f"Ca: {test_simulation['Ca']}")
    # print(f"A: {test_simulation['A']}")

    # Tests of the plotting process
    plot_compartment((a_time_array, some_compartments_results), download=False)
