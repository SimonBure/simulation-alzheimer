import numpy as np
from matplotlib import pyplot as plt

def hill_fct_migration(concentration: float, coefficients: tuple) -> float:
    """
    Hill function to calculate the rate of migration of monomers between cytoplasm and perinuclear crown (PC) or PC and
    nucleus.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM dimers in the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n
    :return: float representing the rate of migration, in h⁻¹.
    """
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    return (b * a ** n) / (a ** n + concentration ** n)


def hill_fct_dimer_formation(concentration: float, coefficients: tuple) -> float:
    """
    Hill function to calculate the rate of dimers formation in the PC.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM-ApoE complexes in
        the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n
    :return: float representing the rate of dimerisation, in h⁻¹.
    """
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    return (a * concentration ** n) / (b ** n + concentration ** n)


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


def compartmental_simulation(duration: float, time_step: float = 1 / 60) -> dict:
    """
    Simulate the evolution of the Alzheimer compartmental model for the given duration.
    :param duration: float giving the time when the simulation must stop, given in hours.
    :param time_step: float representing the time step between 2 computations, given in hours. Default to 1 minute.
    :return: dictionary containing the data arrays of all the 7 compartments.
    """
    # Coefficients definition
    lam = 15  # constant source of ATM dimers in cytoplasm
    d0 = 0.05  # rate of ATM dimers degradation
    d1 = 0.3  # rate of ATM monomers degradation in the nucleus
    k1 = 0.01  # rate of ATM monomers re-dimerisation
    migration_pc_coefs = (400, 0.4, 15)  # parameters for the hill function k2
    # Migration rate of the ATM monomers from the cytoplasm to the PC
    k2 = lambda concentration: hill_fct_dimer_formation(concentration, migration_pc_coefs)  # lighten code with lambda
    migration_nuc_coefs = (80, 0.5, 5)
    # Migration rate of the ATM monomers from the PC to the nucleus
    k3 = lambda concentration: hill_fct_dimer_formation(concentration, migration_nuc_coefs)
    k4 = 0.05  # rate of ATM-ApoE complexes formation
    cplx_formation_coefs = (0.4, 150, 15)
    # Dimerisation rate of the ATM monomers inside the PC
    k5 = lambda concentration: hill_fct_dimer_formation(concentration, cplx_formation_coefs)
    cs = 0.002  # rate of protein dissociation during constant stress
    irradiation_coefs = (0.8, 9, 2)  # parameters for the gaussian representation of the irradiation stress

    # Initial conditions
    dc = 300  # ATM dimers concentration in cytoplasm
    mc = 0  # ATM monomers concentration in cytoplasm
    ma = 0  # ATM monomers concentration in PC
    mn = 0  # ATM monomers concentration in nucleus
    a = 200  # ApoE concentration around the nucleus
    ca = 0  # ATM-ApoE complex concentration in PC
    da = 0  # ATM dimers concentration in PC

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
        # Update rule using Euler explicit numerical scheme
        dc += time_step * (lam - d0 * dc + 0.5 * k1 * mc ** 2)
        mc += time_step * (- k1 * mc ** 2 - k2(da) * mc)
        ma += time_step * (k2(da) * mc - k3(da) * ma - k4 * a * ma - k5(ca) * ma ** 2)
        mn += time_step * (k3(da) * ma - d1 * mn)
        a += time_step * (- k4 * ma * a)
        ca += time_step * (k4 * ma * a)
        da += time_step * (0.5 * k5(ca) * ma ** 2)

        time_simu += time_step

        # Array filling
        dc_array.append(dc)
        mc_array.append(mc)
        ma_array.append(ma)
        mn_array.append(mn)
        a_array.append(a)
        ca_array.append(ca)
        da_array.append(da)

    return {'Dc': dc_array,
            'Mc': mc_array,
            'Ma': ma_array,
            'Mn': mn_array,
            'A': a_array,
            'Ca': ca_array,
            'Da': da_array}


if __name__ == "__main__":
    # Unit test for the different Hill functions
    coef1 = (3, 1, 2)
    concentration1 = 2
    assert hill_fct_migration(concentration1, coef1) == 9 / 13
    assert hill_fct_migration(concentration1, coef1) - 12 / 5 < 1e-5

    # Tests of the compartmental simulations
    test_simulation = compartmental_simulation(10, time_step=2)
    print(test_simulation["Da"])
