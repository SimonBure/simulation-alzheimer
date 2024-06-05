import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt


def migration_monomers(concentration: float, coefficients: tuple, drugs: tuple = (0, 0),
                       destination: str = None) -> float:
    """
    Function to calculate the rate of migration of monomers between cytoplasm and perinuclear crown (PC) or PC and
    nucleus.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM dimers in the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n.
        If antioxidant is True, must also contain the coefficient e of the antioxidant.
    :param drugs: tuple containing the doses of antioxidant and of statin taken. Default to (0, 0).
    :param destination: string indicating the destination of the monomers migrating. Must be 'pc' or 'nucleus'.
        Abbreviations 'n' or 'nucl' are permitted for the nucleus.
    :return: float representing the rate of migration, in h⁻¹.
    """
    # Normal migration rate
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    migration_rate = (b * a ** n) / (a ** n + concentration ** n)
    e = coefficients[3]  # antioxidant effect
    f = coefficients[4]  # statin effect

    antioxidant = drugs[0]
    statin = drugs[1]

    # Allowing abbreviations for the migration in the nucleus
    writing_possibilities_nucleus = ('nucleus', 'n', 'nucl')
    if destination in writing_possibilities_nucleus:
        migration_rate *= (1 + e * antioxidant) * (1 + f * statin)

    elif destination.upper() == 'PC':
        migration_rate *= 1 / (1 + e * antioxidant)

    return migration_rate


def dimers_formation(concentration: float, coefficients: tuple, antioxidant: float = 0) -> float:
    """
    Hill function to calculate the rate of dimers formation in the PC.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM-ApoE complexes in
        the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n.
    :param antioxidant: float indicating the quantity of antioxidant taken.
    :return: float representing the rate of dimers formation, in h⁻¹.
    """
    # Normal rate of dimers formation
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    dimers_formation_rate = (a * concentration ** n) / (b ** n + concentration ** n)

    # Antioxidant effect
    e = coefficients[3]
    dimers_formation_rate *= 1 / (1 + e * antioxidant)

    return dimers_formation_rate


def irradiation_stress(time: float, coefficients: tuple) -> float:
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


def antioxidant_effect(time: float, coefficients: tuple) -> float:
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


def statin_effect(time: float, coefficients: tuple) -> float:
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


def compartmental_simulation(duration: float, time_step: float = 1 / 60, initial: str = 'nf',
                             experimental: tuple = (False, 'none', 'none')) -> dict:
    """
    Simulate the evolution of the Alzheimer compartmental model for the given duration.
    :param duration: float giving the time when the simulation must stop, given in hours.
    :param time_step: float representing the time step between 2 computations, given in hours. Default to 1 minute.
    :param initial: str indicating which initial condition to use. Can be either 'formed', 'f' if you want the
        perinuclear crown to be formed initially, 'not formed', 'nf', 'no', 'n' if not. You can also give a tuple
        containing the initial conditions if you want to use customized initial conditions.
    :param experimental: tuple containing the indications on the experimental conditions. Elements are the following
        1. irradiation: boolean indicating whether to use the irradiation or not. Default to False.
        2. antioxidant_use: string indicating the way antioxidant are used. Could be 'none', 'no', 'n' if no
            antioxidant are used. Can also be 'constant', 'const', 'cst', 'c' if the dose is constant or 'variable', 'var',
            'v' if the dose vary with the time of the simulation. Default to 'none'.
        3. statin_use: string indicating the way statin are used. Could be 'none', 'no', 'n' if no
        antioxidant are used. Can also be 'constant', 'const', 'cst', 'c' if the dose is constant or 'variable', 'var',
         'v' if the dose vary with the time of the simulation. Default to 'none'.
        4. stress: boolean indicating whether the cell experiments stress or not
    :return: dictionary containing the data arrays of all the 7 compartments.
    """
    crown_formed = ('formed', 'f')
    crown_not_formed = ('not formed', 'nf', 'no', 'n')

    if type(initial) is str and initial not in crown_formed and initial not in crown_not_formed:
        raise ValueError(f"Parameter [initial crown] must be one of the following: {crown_formed} or {crown_not_formed}."
                         f"Input value: {initial}")

    irradiation, antioxidant_use, statin_use, stress = experimental

    # Coefficients definition
    lam = 15  # constant source of ATM dimers in cytoplasm
    d0 = 0.05  # rate of ATM dimers degradation
    d1 = 0.3  # rate of ATM monomers degradation in the nucleus
    a1 = 0.01  # rate of ATM monomers re-dimerisation
    e0 = e1 = e2 = e4 = e5 = 20  # inhibiting impact of the antioxidant
    e3 = e6 = 0.5  # promoting impact of the antioxidant
    f3 = 1  # impact of statin on nucleus permeability
    migration_pc_coefs = (400, 0.4, 15, e2, f3)  # parameters for the hill function k2
    migration_nuc_coefs = (80, 0.5, 5, e3, f3)  # parameters for the hill function k3
    a4 = 0.05  # rate of ATM-ApoE complexes formation
    cplx_formation_coefs = (0.4, 150, 15, e5)
    cs = 0.002 if stress else 0  # rate of protein dissociation during constant stress
    cox = 1  # dose of antioxidant
    sta = 5  # dose of statin
    irradiation_coefs = (0.8, 9, 2)  # parameters for the gaussian representation of the irradiation stress
    antioxidant_coefs = (cox, 9, 3)  # parameters for the gaussian representation of the antioxidant effect
    statin_coefs = (sta, 9, 4)

    # Initial conditions
    if type(initial) is str:
        dc = 300  # ATM dimers in cytoplasm
        mc = 0  # ATM monomers in cytoplasm
        ma = 0  # ATM monomers in PC
        mn = 0  # ATM monomers in nucleus
        a = 200 if initial in crown_not_formed else 0  # ApoE proteins in PC
        ca = 200 if initial in crown_formed else 0  # ATM-ApoE complexes in PC
        da = 300 if initial in crown_formed else 0  # ATM dimers in PC
    elif type(initial) is tuple:
        dc, mc, ma, mn, a, ca, da = initial

    initial_conditions = {'Dc0': dc, 'Mc0': mc, 'Ma0': ma, 'Mn0': mn, 'A0': a, 'Ca0': ca, 'Da0': da}

    # Array for storing data of the simulation
    dc_array = [dc]
    mc_array = [mc]
    ma_array = [ma]
    mn_array = [mn]
    a_array = [a]
    ca_array = [ca]
    da_array = [da]

    time_simu = 0

    # Simulation
    while time_simu < duration:
        time_simu += time_step

        # antioxidant dosage
        no_antioxidant = ('none', 'no', '0', 'n')  # string possibilities for not using antioxidant
        cst_antioxidant = ('constant', 'const', 'cst', 'c')  # string possibilities for using a constant dose
        var_antioxidant = ('variable', 'var', 'v')  # string possibilities for using a dynamic dose
        if antioxidant_use in no_antioxidant:
            antioxidant = 0
        elif antioxidant_use in cst_antioxidant:
            antioxidant = cox
        elif antioxidant_use in var_antioxidant:
            antioxidant = antioxidant_effect(time_simu, antioxidant_coefs)
        else:
            raise ValueError("Please select a valid use for the antioxidant")

        # Statin dosage
        no_statin = ('none', 'no', '0', 'n')  # string possibilities for not using antioxidant
        cst_statin = ('constant', 'const', 'cst', 'c')  # string possibilities for using a constant dose
        var_statin = ('variable', 'var', 'v')  # string possibilities for using a dynamic dose
        if statin_use in no_statin:
            statin = 0
        elif statin_use in cst_statin:
            statin = sta
        elif statin_use in var_statin:
            statin = statin_effect(time_simu, statin_coefs)
        else:
            raise ValueError("Please select a valid use for the statin")

        drugs = (antioxidant, statin)  # tuple of the doses

        # Computing dynamic rates
        k1 = a1 / (1 + e1 * antioxidant)  # total rate of dimers formation
        # Migration rate of the ATM monomers from cytoplasm into the PC
        k2 = migration_monomers(da, migration_pc_coefs, drugs, 'pc')
        # Migration rate of the ATM monomers from PC into the nucleus
        k3 = migration_monomers(da, migration_nuc_coefs, drugs, 'nucleus')
        k4 = a4 / (1 + e4 * antioxidant)
        # Dimers formation  rate of the ATM monomers inside the PC
        k5 = dimers_formation(ca, cplx_formation_coefs, antioxidant)
        k6 = e6 * antioxidant  # dispersion caused by the antioxidant
        # Stress factor: constant stress + irradiation + antioxidant effects
        g = cs / (1 + e0 * antioxidant)
        if irradiation:
            g += irradiation_stress(time_simu, irradiation_coefs)

        params = {'lam': lam, 'd0': d0, 'd1': d1, 'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'g': g}

        # Update rule using Euler explicit numerical scheme
        dc += time_step * (lam - d0 * dc + (0.5 * k1 * mc ** 2) - g * dc)
        mc += time_step * ((-k1 * mc ** 2) - k2 * mc + k6 * ma + 2 * g * dc)
        ma += time_step * (k2 * mc - k3 * ma - k4 * a * ma - (k5 * ma ** 2) - k6 * ma + 2 * g * da + g * ca)
        mn += time_step * (k3 * ma - d1 * mn)
        a += time_step * (-k4 * ma * a + g * ca)
        ca += time_step * (k4 * ma * a - g * ca)
        da += time_step * ((0.5 * k5 * ma ** 2) - g * da)

        # Array filling
        dc_array.append(dc)
        mc_array.append(mc)
        ma_array.append(ma)
        mn_array.append(mn)
        a_array.append(a)
        ca_array.append(ca)
        da_array.append(da)

    time_array = np.linspace(0, duration, num=len(dc_array))

    # Equilibrium
    print(f"Fixed points without stress: {get_fixed_points(params, initial_conditions, False)}")
    fixed_point_stress = get_fixed_points(params, initial_conditions, True)
    print(f"Fixed points with stress: {fixed_point_stress[0]}\n{fixed_point_stress[1]}")
    get_existence_conditions(params)

    return {'Dc': dc_array,
            'Mc': mc_array,
            'Ma': ma_array,
            'Mn': mn_array,
            'A': a_array,
            'Ca': ca_array,
            'Da': da_array,
            'Time': time_array}


def plot_cyto_nucleus(data_compartments: dict, ax: matplotlib.figure.Axes) -> matplotlib.figure.Axes:
    """
    Function to plot the trajectories of the simulation in the cytoplasm and the nucleus.
    :param data_compartments: dict containing the results of the simulation.
    :param ax: matplotlib.figure.Axes object passed to the function to add the trajectories on it for further display.
    :return: matplotlib.figure.Axes object with trajectories of the simulation in the cytoplasm and the nucleus added.
    """
    # Data plotting
    ax.plot(data_compartments['Time'], data_compartments['Dc'], label='Dc', color='blue')
    ax.plot(data_compartments['Time'], data_compartments['Mc'], label='Mc', color='green')
    ax.plot(data_compartments['Time'], data_compartments['Mn'], label='Mn', color='red')

    # Titles & labels
    ax.set_title("Populations evolution in the cytoplasm & nucleus}")
    ax.set_xlabel('Time (h)', fontsize=12)
    ax.set_ylabel('Populations', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    return ax


def plot_perinuclear_crown(data_compartments: dict, ax: matplotlib.figure.Axes) -> matplotlib.figure.Axes:
    """
    Function to plot the trajectories of the simulation in the perinuclear crown.
    :param data_compartments: dict containing the results of the simulation.
    :param ax: matplotlib.figure.Axes object passed to the function to add the trajectories on it for further display.
    :return: matplotlib.figure.Axes object with trajectories of the simulation in the perinuclear crown added.
    """
    # Data plotting
    ax.plot(data_compartments['Time'], data_compartments['A'], label='A', color='magenta')
    ax.plot(data_compartments['Time'], data_compartments['Ca'], label='Ca', color='darkorange')
    ax.plot(data_compartments['Time'], data_compartments['Da'], label='Da', color='crimson')
    ax.plot(data_compartments['Time'], data_compartments['Ma'], label='Ma', color='dodgerblue')

    # Titles & labels
    ax.set_title("Populations evolution in the perinuclear crown")
    ax.set_xlabel('Time (h)', fontsize=12)
    ax.set_ylabel('Populations', fontsize=12)
    ax.legend(loc='upper left',  bbox_to_anchor=(1, 1), fontsize=12)

    return ax


def plot_compartment(data_compartments: dict, download: bool = False):
    """
    Function of higher level to specify which compartment to plot, a specific title and whether to download the plot or
    not.
    :param data_compartments: dict containing the results of the simulation.
    :param download: boolean indicating whether to download the plot or not.
    """
    fig, (ax_nucl, ax_pc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_nucl = plot_cyto_nucleus(data_compartments, ax_nucl)
    ax_pc = plot_perinuclear_crown(data_compartments, ax_pc)

    plt.subplots_adjust(wspace=0.4)  # setting horizontal space between the two plots

    if download is True:
        fig.savefig(f'FIGURES/plot.png')

    plt.show()


def get_discriminant(parameters: dict) -> float:
    """
    Function to compute the value of the discriminant needed to get the value of the equilibrium with stress.
    :param parameters: dict containing the parameters needed to compute the discriminant.
    :return: the discriminant value as a float value.
    """
    discriminant = (((parameters['k3'] - 2 * parameters['k6']) * (parameters['d0'] + parameters['g']))**2 +
                    8 * parameters['k1'] * parameters['d0'] * parameters['g'] * parameters['lam'] *
                    ((parameters['k3'] - parameters['k6']) / parameters['k2'])**2)
    return discriminant


def get_fixed_points(parameters: dict, init: dict, stress: bool) -> tuple[dict, dict]:
    """
    Function to compute the theoretical values of the equilibrium points. I
    :param parameters: dict containing the parameters needed to compute the equilibrium.
    :param init: dict of initial values needed to compute the equilibrium.
    :param stress: bool indicating whether to compute the equilibrium with or without stress.
    :return: If stress is True then return two dict, each containing a value for the equilibrium. If stress is False
        then return only one dict of the simple equilibrium.
    """
    if stress:
        disc = get_discriminant(parameters)
        ma_eq_plus = parameters['k2']**2 * (2 * parameters['k6'] - parameters['k3'] + np.sqrt(disc)) / (2 * parameters['k1'] * parameters['d0'] * (parameters['k3'] - parameters['k6'])**2)
        ma_eq_neg = parameters['k2']**2 * (2 * parameters['k6'] - parameters['k3'] - np.sqrt(disc)) / (2 * parameters['k1'] * parameters['d0'] * (parameters['k3'] - parameters['k6'])**2)
        mc_eq_plus = ((parameters['k3'] - parameters['k6']) / parameters['k2']) * ma_eq_plus
        mc_eq_neg = ((parameters['k3'] - parameters['k6']) / parameters['k2']) * ma_eq_neg
        mn_eq_plus = (parameters['k3'] / parameters['d1']) * ma_eq_plus
        mn_eq_neg = (parameters['k3'] / parameters['d1']) * ma_eq_neg
        da_eq_plus = (parameters['k5'] / 2 * parameters['g']) * ma_eq_plus**2
        da_eq_neg = (parameters['k5'] / 2 * parameters['g']) * ma_eq_neg**2
        dc_eq_plus = (parameters['lam'] + (parameters['k1'] / 2) * mc_eq_plus**2) / (parameters['d0'] + parameters['g'])
        dc_eq_neg = (parameters['lam'] + (parameters['k1'] / 2) * mc_eq_neg ** 2) / (parameters['d0'] + parameters['g'])
        a_eq_plus = parameters['g'] * (init['A0'] + init['Ca0']) / (parameters['g'] + parameters['k4'] * ma_eq_plus)
        a_eq_neg = parameters['g'] * (init['A0'] + init['Ca0']) / (parameters['g'] + parameters['k4'] * ma_eq_neg)
        ca_eq_plus = init['A0'] + init['Ca0'] - a_eq_plus
        ca_eq_neg = init['A0'] + init['Ca0'] - a_eq_neg
        eq_plus = {'Dc': dc_eq_plus, 'Mc': mc_eq_plus, 'Ma': ma_eq_plus, 'Mn': mn_eq_plus, 'A': a_eq_plus,
                   'Ca': ca_eq_plus, 'Da': da_eq_plus}
        eq_neg = {'Dc': dc_eq_neg, 'Mc': mc_eq_neg, 'Ma': ma_eq_neg, 'Mn': mn_eq_neg, 'A': a_eq_neg, 'Ca': ca_eq_neg,
                  'Da': da_eq_neg}
        equilibrium = eq_plus, eq_neg
    else:
        equilibrium = {'Dc': parameters['lam'] / parameters['d0'], 'Mc': 0, 'Ma': 0, 'Mn': 0, 'A': 'A*', 'Ca': 'Ca*',
                       'Da': 'Da*'}
    return equilibrium


def get_existence_conditions(parameters: dict):
    """
    Function to write the theoretical conditions on the parameters that are needed for the equilibria to exist.
    :param parameters: dict containing the parameters needed to verify the conditions.
    """
    disc = get_discriminant(parameters)
    print("Ma+ exists if 2k6 + np.sqrt(Δ) > k3")
    print(f"Condition fulfilled ? {2 * parameters['k6'] + np.sqrt(disc) > parameters['k3']}")
    print("Ma- exists if 2k6 > np.sqrt(Δ) + k3")
    print(f"Condition fulfilled ? {2 * parameters['k6'] > np.sqrt(disc) + parameters['k3']}")


if __name__ == "__main__":
    # Plotting parameter functions
    # k2 - k3
    # concentration = np.linspace(0, 1000, num=1000)
    # plt.plot(concentration, hill_fct_migration(concentration, coef2))
    # plt.show()

    # k5
    # coef3 = (0.4, 150, 15)
    # concentration = np.linspace(0, 200, num=200)
    # plt.plot(concentration, hill_fct_dimer_formation(concentration, coef3), color="red")
    # plt.show()

    # g
    # coef4 = (0.8, 9, 2)
    # time = np.linspace(0, 10, num=100)
    # g_values = irradiation_stress(time, coef4)
    # plt.plot(time, g_values)
    # plt.show()

    # antioxidant
    # coef4 = (1, 9, 3)
    # time = np.linspace(0, 20, num=100)
    # aox_values = antioxidant_effect(time, coef4)
    # plt.plot(time, aox_values)
    # plt.show()

    # Compartmental simulations
    # is_irradiated = True
    is_irradiated = False
    antioxidant_dose = 'no'
    # antioxidant_dose = 'cst'
    # antioxidant_dose = 'var'
    statin_dose = 'no'
    # statin_dose = 'cst'
    # statin_dose = 'var'
    # is_stress = True
    is_stress = False
    experimental_conditions = (is_irradiated, antioxidant_dose, statin_dose, is_stress)
    # np.random.random
    # initial_conditions = (150, 150, 150, 150, 150, 150, 150)
    # initial_conditions = 'formed'
    # initial_conditions = 'not formed'
    simulation = compartmental_simulation(24, 1 / 60, initial=initial_conditions,
                                          experimental=experimental_conditions)
    # print(f"Da: {test_simulation['Da']}")
    # print(f"Ma: {test_simulation['Ma']}")
    # print(f"Ca: {test_simulation['Ca']}")
    # print(f"A: {test_simulation['A']}")

    # Tests of the plotting process
    plot_compartment(simulation, download=True)

    # Equilibrium verification
