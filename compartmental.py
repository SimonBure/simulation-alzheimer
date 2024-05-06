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
        If antioxidants is True, must also contain the coefficient e of the antioxidants.
    :param drugs: tuple containing the doses of antioxidants and of statins taken. Default to (0, 0).
    :param destination: string indicating the destination of the monomers migrating. Must be 'pc' or 'nucleus'.
        Abbreviations 'n' or 'nucl' are permitted for the nucleus.
    :return: float representing the rate of migration, in h⁻¹.
    """
    # Normal migration rate
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    migration_rate = (b * a ** n) / (a ** n + concentration ** n)
    e = coefficients[3]  # antioxidants effect
    f = coefficients[4]  # statins effect

    antioxidants = drugs[0]
    statins = drugs[1]

    # Allowing abbreviations for the migration in the nucleus
    writing_possibilities_nucleus = ('nucleus', 'n', 'nucl')
    if destination in writing_possibilities_nucleus:
        migration_rate *= (1 + e * antioxidants) * (1 + f * statins)

    elif destination.upper() == 'PC':
        migration_rate *= 1 / (1 + e * antioxidants)

    return migration_rate


def dimers_formation(concentration: float, coefficients: tuple, antioxidants: float = 0) -> float:
    """
    Hill function to calculate the rate of dimers formation in the PC.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM-ApoE complexes in
        the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n.
    :param antioxidants: float indicating the quantity of antioxidants taken.
    :return: float representing the rate of dimers formation, in h⁻¹.
    """
    # Normal rate of dimers formation
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    dimers_formation_rate = (a * concentration ** n) / (b ** n + concentration ** n)

    # Antioxidant effect
    e = coefficients[3]
    dimers_formation_rate *= 1 / (1 + e * antioxidants)

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


def antioxidants_effect(time: float, coefficients: tuple) -> float:
    """
    Calculates the value of the antioxidants effects using a Gaussian function.
    :param time: float giving the actual time of the simulation.
    :param coefficients: tuple of 3 elements containing the coefficient of the Gaussian function. Must be in order
        c_aox, m_aox, sigma_aox.
    :return: float representing the value of the antioxidants effects.
    """
    c_aox = coefficients[0]  # antioxidants dose
    m_aox = coefficients[1]  # time when the antioxidants are given
    sigma_aox = coefficients[2]  # rapidity of the antioxidants effects

    return c_aox * np.exp(-(time - m_aox) ** 2 / (2 * sigma_aox ** 2))


def statins_effect(time: float, coefficients: tuple) -> float:
    """
    Calculates the value of the statins effects using a Gaussian function.
    :param time: float giving the actual time of the simulation.
    :param coefficients: tuple of 3 elements containing the coefficient of the Gaussian function. Must be in order
            s_ta, m_ta, sigma_ta.
    :return: float representing the value of the statins effects.
    """
    s_ta = coefficients[0]  # statins dose
    m_ta = coefficients[1]  # time when the statins are taken
    sigma_ta = coefficients[2]  # rapidity of the statins effects or biocompatibility

    return s_ta * np.exp(-(time - m_ta) ** 2 / (2 * sigma_ta ** 2))


def compartmental_simulation(duration: float, time_step: float = 1 / 60, initial_crown: str = 'nf',
                             experimental: tuple = (False, 'none', 'none')) -> dict:
    """
    Simulate the evolution of the Alzheimer compartmental model for the given duration.
    :param duration: float giving the time when the simulation must stop, given in hours.
    :param time_step: float representing the time step between 2 computations, given in hours. Default to 1 minute.
    :param initial_crown: string indicating which initial condition to use. Can be either 'formed', 'f' if you want the
        perinuclear crown to be formed initially or 'not formed', 'nf', 'no', 'n' if not.
    :param experimental: tuple containing the indications on the experimental conditions. Elements are the following
        1. irradiation: boolean indicating whether to use the irradiation or not. Default to False.
        2. antioxidants_use: string indicating the way antioxidants are used. Could be 'none', 'no', 'n' if no
            antioxidants are used. Can also be 'constant', 'const', 'cst', 'c' if the dose is constant or 'variable', 'var',
            'v' if the dose vary with the time of the simulation. Default to 'none'.
        3. statins_use: string indicating the way statins are used. Could be 'none', 'no', 'n' if no
        antioxidants are used. Can also be 'constant', 'const', 'cst', 'c' if the dose is constant or 'variable', 'var',
         'v' if the dose vary with the time of the simulation. Default to 'none'.
    :return: dictionary containing the data arrays of all the 7 compartments.
    """
    crown_formed = ('formed', 'f')
    crown_not_formed = ('not formed', 'nf', 'no', 'n')

    if initial_crown not in crown_not_formed and initial_crown not in crown_not_formed:
        raise ValueError(f"Parameter [initial crown] must be one of the following: {crown_formed} or {crown_not_formed}."
                         f"Input value: {initial_crown}")

    irradiation, antioxidants_use, statins_use = experimental

    # Coefficients definition
    lam = 15  # constant source of ATM dimers in cytoplasm
    d0 = 0.05  # rate of ATM dimers degradation
    d1 = 0.3  # rate of ATM monomers degradation in the nucleus
    a1 = 0.01  # rate of ATM monomers re-dimerisation
    e0 = e1 = e2 = e4 = e5 = 20  # inhibiting impact of the antioxidants
    e3 = e6 = 0.5  # promoting impact of the antioxidants
    f3 = 1  # impact of statins on nucleus permeability
    migration_pc_coefs = (400, 0.4, 15, e2, f3)  # parameters for the hill function k2
    migration_nuc_coefs = (80, 0.5, 5, e3, f3)  # parameters for the hill function k3
    a4 = 0.05  # rate of ATM-ApoE complexes formation
    cplx_formation_coefs = (0.4, 150, 15, e5)
    cs = 0.002  # rate of protein dissociation during constant stress
    cox = 1  # dose of antioxidants
    sta = 5  # dose of statins
    irradiation_coefs = (0.8, 9, 2)  # parameters for the gaussian representation of the irradiation stress
    antioxidants_coefs = (cox, 9, 3)  # parameters for the gaussian representation of the antioxidants effect
    statins_coefs = (sta, 9, 4)

    # Initial conditions
    dc = 300  # ATM dimers concentration in cytoplasm
    a = 0 if initial_crown in crown_formed else 200  # ApoE proteins concentration in PC
    ca = 200 if initial_crown in crown_not_formed else 0  #
    da = 300 if initial_crown in crown_not_formed else 0
    mc = 0  # ATM monomers concentration in cytoplasm
    ma = 0  # ATM monomers concentration in PC
    mn = 0  # ATM monomers concentration in nucleus

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

        # Antioxidants dosage
        no_antioxidants = ('none', 'no', '0', 'n')  # string possibilities for not using antioxidants
        cst_antioxidants = ('constant', 'const', 'cst', 'c')  # string possibilities for using a constant dose
        var_antioxidants = ('variable', 'var', 'v')  # string possibilities for using a dynamic dose
        if antioxidants_use in no_antioxidants:
            antioxidants = 0
        elif antioxidants_use in cst_antioxidants:
            antioxidants = cox
        elif antioxidants_use in var_antioxidants:
            antioxidants = antioxidants_effect(time_simu, antioxidants_coefs)
        else:
            raise ValueError("Please select a valid use for the antioxidants")

        # Statins dosage
        no_statins = ('none', 'no', '0', 'n')  # string possibilities for not using antioxidants
        cst_statins = ('constant', 'const', 'cst', 'c')  # string possibilities for using a constant dose
        var_statins = ('variable', 'var', 'v')  # string possibilities for using a dynamic dose
        if statins_use in no_statins:
            statins = 0
        elif statins_use in cst_statins:
            statins = sta
        elif statins_use in var_statins:
            statins = statins_effect(time_simu, statins_coefs)
        else:
            raise ValueError("Please select a valid use for the statins")

        drugs = (antioxidants, statins)  # tuple of the doses

        # Computing dynamic rates
        k1 = a1 / (1 + e1 * antioxidants)  # total rate of dimers formation
        # Migration rate of the ATM monomers from cytoplasm into the PC
        k2 = migration_monomers(da, migration_pc_coefs, drugs, 'pc')
        # Migration rate of the ATM monomers from PC into the nucleus
        k3 = migration_monomers(da, migration_nuc_coefs, drugs, 'nucleus')
        k4 = a4 / (1 + e4 * antioxidants)
        # Dimers formation  rate of the ATM monomers inside the PC
        k5 = dimers_formation(ca, cplx_formation_coefs, antioxidants)
        k6 = e6 * antioxidants  # dispersion caused by the antioxidants
        # Stress factor: constant stress + irradiation + antioxidants effects
        g = cs / (1 + e0 * antioxidants)
        if irradiation:
            g += irradiation_stress(time_simu, irradiation_coefs)

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

    return {'Dc': dc_array,
            'Mc': mc_array,
            'Ma': ma_array,
            'Mn': mn_array,
            'A': a_array,
            'Ca': ca_array,
            'Da': da_array,
            'Time': time_array}


def plot_cyto_nucleus(data_compartments: dict, title: str, ax: matplotlib.figure.Axes) -> matplotlib.figure.Axes:
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
    ax.set_title(f"Populations evolution in the cytoplasm & nucleus\n{title}")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Populations')
    ax.legend(loc='upper right')

    return ax


def plot_perinuclear_crown(data_compartments: dict, title: str, ax: matplotlib.figure.Axes) -> matplotlib.figure.Axes:
    """
    Function to plot the trajectories of the simulation in the perinuclear crown.
    :param data_compartments: dict containing the results of the simulation.
    :param title: Additional content to be added to the title of the plot.
    :param ax: matplotlib.figure.Axes object passed to the function to add the trajectories on it for further display.
    :return: matplotlib.figure.Axes object with trajectories of the simulation in the perinuclear crown added.
    """
    # Data plotting
    ax.plot(data_compartments['Time'], data_compartments['A'], label='A', color='magenta')
    ax.plot(data_compartments['Time'], data_compartments['Ca'], label='Ca', color='y')
    ax.plot(data_compartments['Time'], data_compartments['Da'], label='Da', color='turquoise')
    ax.plot(data_compartments['Time'], data_compartments['Ma'], label='Ma', color='purple')

    # Titles & labels
    ax.set_title(f"Populations evolution in the perinuclear crown\n{title}")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Populations')
    ax.legend(loc='upper right')

    return ax


def plot_compartment(data_compartments: dict, title: str = None, download: bool = False):
    """
    Function of higher level to specify which compartment to plot, a specific title and whether to download the plot or
    not.
    :param data_compartments: dict containing the results of the simulation.
    :param title: optional string containing a more detailed or specific title for the plots.
    :param download: boolean indicating whether to download the plot or not.
    """
    fig, (ax_nucl, ax_pc) = plt.subplots(1, 2, figsize=(10, 5))

    ax_nucl = plot_cyto_nucleus(data_compartments, title, ax_nucl)
    ax_pc = plot_perinuclear_crown(data_compartments, title, ax_pc)

    if download is True:
        fig.savefig(f'FIGURES/plot.png')

    plt.show()


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

    # Antioxidants
    # coef4 = (1, 9, 3)
    # time = np.linspace(0, 20, num=100)
    # aox_values = antioxidants_effect(time, coef4)
    # plt.plot(time, aox_values)
    # plt.show()

    # Tests of the compartmental simulations
    is_irradiated = True
    antioxidants_dose = 'var'
    statins_dose = 'var'
    experimental_conditions = (irradiation, antioxidants_use, statins_use)
    test_simulation = compartmental_simulation(24, 1 / 60, initial_crown='formed',
                                               experimental=experimental_conditions)
    # print(f"Da: {test_simulation['Da']}")
    # print(f"Ma: {test_simulation['Ma']}")
    # print(f"Ca: {test_simulation['Ca']}")
    # print(f"A: {test_simulation['A']}")

    # Tests of the plotting process
    plot_compartment(test_simulation, title='irradiation & antioxidants & statins at 9h', download=True)
