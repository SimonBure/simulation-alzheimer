import compartmental as ct
import random as rdm


def study_stability_without_stress(experimental_conditions):
    _, __, ___, equilibrium_without_stress = ct.compartmental_simulation(24, 1 / 60, 'not formed',
                                                                         experimental_conditions)
    print(f"Equilibrium value: {equilibrium_without_stress}")

    neighborhood_above_equilibrium_without_stress = get_neighborhood_above_equilibrium_without_stress(equilibrium_without_stress)

    time_array_above, stability_above_results, _, __ = ct.compartmental_simulation(2400, 1 / 60,
                                                                                   neighborhood_above_equilibrium_without_stress,
                                                                                   experimental_conditions)

    ct.plot_compartment((time_array_above, stability_above_results), download=False)

    neighborhood_below_equilibrium_without_stress = get_neighborhood_below_equilibrium_without_stress(equilibrium_without_stress)

    time_array_below, stability_below_results, _, __ = ct.compartmental_simulation(2400, 1 / 60,
                                                                                   neighborhood_below_equilibrium_without_stress,
                                                                                   experimental_conditions)

    ct.plot_compartment((time_array_below, stability_below_results), download=False)


def get_neighborhood_above_equilibrium_without_stress(equilibrium: dict) -> tuple:
    dc_init = equilibrium['Dc'] + rdm.randint(5, 20)
    mc_init = equilibrium['Mc'] + rdm.randint(5, 20)
    ma_init = equilibrium['Ma'] + rdm.randint(5, 20)
    mn_init = equilibrium['Mn'] + rdm.randint(5, 20)
    a_init = rdm.randint(0, 300)
    ca_init = rdm.randint(0, 300)
    da_init = rdm.randint(0, 300)

    return dc_init, mc_init, ma_init, mn_init, a_init, ca_init, da_init


def get_neighborhood_below_equilibrium_without_stress(equilibrium: dict) -> tuple:
    dc_init = equilibrium['Dc'] - rdm.randint(5, 20)
    mc_init = 0
    ma_init = 0
    mn_init = 0
    a_init = rdm.randint(0, 300)
    ca_init = rdm.randint(0, 300)
    da_init = rdm.randint(0, 300)

    return dc_init, mc_init, ma_init, mn_init, a_init, ca_init, da_init


def study_stability_with_stress(experimental_conditions):
    _, __, equilibrium_with_stress, ___ = ct.compartmental_simulation(24, 1 / 60, 'not formed',
                                                                      experimental_conditions)
    print(f"Equilibrium value: {equilibrium_with_stress}")
    equilibrium_positive = equilibrium_with_stress[0]
    equilibrium_negative = equilibrium_with_stress[1]

    neighborhood_above_equilibrium_with_stress = get_neighborhood_above_equilibrium_with_stress(equilibrium_positive)
    print(neighborhood_above_equilibrium_with_stress)

    time_array_above, stability_above_results, _, __ = ct.compartmental_simulation(2400, 1 / 60,
                                                                                   neighborhood_above_equilibrium_with_stress,
                                                                                   experimental_conditions)

    ct.plot_compartment((time_array_above, stability_above_results), download=False)

    neighborhood_below_equilibrium_with_stress = get_neighborhood_below_equilibrium_with_stress(equilibrium_with_stress)

    time_array_below, stability_below_results, _, __ = ct.compartmental_simulation(2400, 1 / 60,
                                                                                   neighborhood_below_equilibrium_with_stress,
                                                                                   experimental_conditions)

    ct.plot_compartment((time_array_below, stability_below_results), download=False)


def get_neighborhood_above_equilibrium_with_stress(equilibrium: dict) -> tuple:
    dc_init = equilibrium['Dc'] + rdm.randint(5, 20)
    mc_init = equilibrium['Mc'] + rdm.randint(5, 20)
    ma_init = equilibrium['Ma'] + rdm.randint(5, 20)
    mn_init = equilibrium['Mn'] + rdm.randint(5, 20)
    a_init = equilibrium['A'] + rdm.randint(5, 20)
    ca_init = equilibrium['Ca'] + rdm.randint(5, 20)
    da_init = equilibrium['Da'] + rdm.randint(5, 20)

    return dc_init, mc_init, ma_init, mn_init, a_init, ca_init, da_init


def get_neighborhood_below_equilibrium_with_stress(equilibrium: dict) -> tuple:
    dc_init = equilibrium['Dc'] - rdm.randint(5, 20)
    mc_init = equilibrium['Mc'] - rdm.randint(5, 20)
    ma_init = equilibrium['Ma'] - rdm.randint(5, 20)
    mn_init = equilibrium['Mn'] - rdm.randint(5, 20)
    a_init = equilibrium['A'] - rdm.randint(5, 20)
    ca_init = equilibrium['Ca'] - rdm.randint(5, 20)
    da_init = equilibrium['Da'] - rdm.randint(5, 20)

    return dc_init, mc_init, ma_init, mn_init, a_init, ca_init, da_init


if __name__ == "__main__":
    some_experimental_conditions_without_stress = (False, 'no', 'no', False)
    study_stability_without_stress(some_experimental_conditions_without_stress)

    some_experimental_conditions_with_stress = (False, 'no', 'no', True)
    study_stability_with_stress(some_experimental_conditions_with_stress)
