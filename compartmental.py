import numpy as np
from matplotlib import pyplot as plt

def hill_fct_migration(concentration: float, coefficients: tuple) -> float:
    """
    Hill function to calculate the rate of migration of monomers between cytoplasm and perinuclear crown (PC) or PC and
    nucleus.
    :param concentration: float and variable of the Hill function. Here it is the concentration of ATM dimers in the PC.
    :param coefficients: tuple of 3 elements containing the coefficient of the Hill function. Must be in order a, b, n
    :return: rate of migration, in h⁻¹.
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
    :return: rate of dimerisation, in h⁻¹.
    """
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    return (a * concentration ** n) / (b ** n + concentration ** n)


def compartmental_simulation(duration: int, step: float = 1e-3) -> dict:
    """
    Simulate the evolution of the Alzheimer compartmental model for the given duration.
    :param duration: int giving the number of steps for the simulation
    :param step: float representing the time step between 2 computations
    :return: dictionary containing the data arrays of all the 7 compartments
    """
    # Coefficients definition
    lam = 15  # constant source of ATM dimers in cytoplasm
    d0 = 0.05  # rate of ATM dimers degradation
    d1 = 0.3  # rate of ATM monomers degradation in the nucleus
    k1 = 0.01  # rate of ATM monomers re-dimerisation
    coef2 = (400, 0.4, 15)  # parameters for the hill function k2
    # Migration rate of the ATM monomers from the cytoplasm to the PC
    k2 = lambda concentration: hill_fct_dimer_formation(concentration, coef2)  # lambda function to lighten the notation
    coef3 = (80, 0.5, 5)
    # Migration rate of the ATM monomers from the PC to the nucleus
    k3 = lambda concentration: hill_fct_dimer_formation(concentration, coef3)
    k4 = 0.05  # rate of ATM-ApoE complexes formation
    coef5 = (0.4, 150, 15)
    # Dimerisation rate of the ATM monomers inside the PC
    k5 = lambda concentration: hill_fct_dimer_formation(concentration, coef5)

    # Initial conditions
    dc = 300  # ATM dimers concentration in cytoplasm
    mc = 0  # ATM monomers concentration in cytoplasm
    ma = 0  # ATM monomers concentration in PC
    mn = 0  # ATM monomers concentration in nucleus
    a = 200  # ApoE concentration around the nucleus
    ca = 0  # ATM-ApoE complex concentration in PC
    da = 0  # ATM dimers concentration in PC

    # Array for storing data of the simulation
    dc_array = np.zeros(duration, dtype='float')
    mc_array = np.zeros(duration, dtype='float')
    ma_array = np.zeros(duration, dtype='float')
    mn_array = np.zeros(duration, dtype='float')
    a_array = np.zeros(duration, dtype='float')
    ca_array = np.zeros(duration, dtype='float')
    da_array = np.zeros(duration, dtype='float')

    # Simulation
    for i in range(duration):
        # Array filling
        dc_array[i] = dc
        mc_array[i] = mc
        ma_array[i] = ma
        mn_array[i] = mn
        a_array[i] = a
        ca_array[i] = ca
        da_array[i] = da

        # Update rule using Euler explicit numerical scheme
        dc += step * (lam - d0 * dc + 0.5 * k1 * mc ** 2)
        mc += step * (- k1 * mc ** 2 - k2(da) * mc)
        ma += step * (k2(da) * mc - k3(da) * ma - k4 * a * ma - k5(ca) * ma ** 2)
        mn += step * (k3(da) * ma - d1 * mn)
        a += step * (- k4 * ma * a)
        ca += step * (k4 * ma * a)
        da += step * (0.5 * k5(ca) * ma ** 2)

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
    test_simulation = compartmental_simulation(10, step=2)
    print(test_simulation["Dc"])
