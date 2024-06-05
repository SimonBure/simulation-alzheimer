import compartmental as ct
import random as rdm


def get_equilibrium_neighborhood(equilibrium: dict, above: bool, stress: bool) -> tuple:
    if above:
        dc_init = equilibrium['Dc'] + rdm.randint(5, 20)
        mc_init = equilibrium['Mc'] + rdm.randint(5, 20)
        ma_init = equilibrium['Ma'] + rdm.randint(5, 20)
        mn_init = equilibrium['Mn'] + rdm.randint(5, 20)
        a_init = equilibrium['A'] + rdm.randint(5, 20) if stress else rdm.randint(0, 300)
        ca_init = equilibrium['Ca'] + rdm.randint(5, 20) if stress else rdm.randint(0, 300)
        da_init = equilibrium['Da'] + rdm.randint(5, 20) if stress else rdm.randint(0, 300)
    else:
        dc_init = equilibrium['Dc'] - rdm.randint(5, 20)
        mc_init = equilibrium['Mc'] - rdm.randint(5, 20) if stress else 0
        ma_init = equilibrium['Ma'] - rdm.randint(5, 20) if stress else 0
        mn_init = equilibrium['Mn'] - rdm.randint(5, 20) if stress else 0
        a_init = equilibrium['A'] - rdm.randint(5, 20) if stress else rdm.randint(0, 300)
        ca_init = equilibrium['Ca'] - rdm.randint(5, 20) if stress else rdm.randint(0, 300)
        da_init = equilibrium['Da'] - rdm.randint(5, 20) if stress else rdm.randint(0, 300)
    return dc_init, mc_init, ma_init, mn_init, a_init, ca_init, da_init


if __name__ == "__main__":
    # TESTING THE STABILITY OF THE SIMPLE EQUILIBRIUM WITHOUT STRESS
    # STABILITY ABOVE THE EQUILIBRIUM
    # Parameters for the compartmental simulation
    is_irradiated = False
    antioxidant_dose = 'no'
    statin_dose = 'no'
    is_stress = False
    initial_conditions = 'not formed'
    experimental_conditions = (is_irradiated, antioxidant_dose, statin_dose, is_stress)

    results, eq_stress, eq_no_stress = ct.compartmental_simulation(24, 1 / 60, initial=initial_conditions,
                                                                   experimental=experimental_conditions)

    ngbh_eq = get_equilibrium_neighborhood(eq_no_stress, False, False)
    print(f"Starting point for the stability study:\n{ngbh_eq}")

    stability_results, _, __ = ct.compartmental_simulation(2400, 1 / 60, initial=ngbh_eq,
                                                           experimental=experimental_conditions)

    ct.plot_compartment(stability_results, download=True)
