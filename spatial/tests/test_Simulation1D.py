from spatial.oneD.Simulation1D import Simulation1D


def test_setup_atm_apoe_system() -> bool:
    a_sim = Simulation1D(1, 5, 1, 5)
    a_sim.setup_atm_apoe_system(1, 1, (1, 1), (1, 1), (1, 1, 1), (1, 1, 1), 1.2)

    return True


def test_setup_experimental_conditions() -> bool:
    a_sim = Simulation1D(1, 5, 1, 5)
    a_sim.setup_atm_apoe_system(1, 1, (1, 1), (1, 1), (1, 1, 1), (1, 1, 1), 1.2)
    a_sim.setup_experimental_conditions((), (), ())
    a_sim.setup_experimental_conditions((0, 1), (5, 6), (8, 9))
    a_sim.setup_experimental_conditions(((0, 1), (33, 55)), ((5, 6), (14, 18)), ((8, 9), (36, 36.5)))
    return True


def test_is_experiment_now() -> bool:
    a_sim = Simulation1D(1, 5, 1, 5)
    a_sim.setup_atm_apoe_system(1, 1, (1, 1), (1, 1), (1, 1, 1), (1, 1, 1), 1.2)
    a_sim.setup_experimental_conditions((0.3, 1), (0.8, 1), (0.4, 0.6))
    a_sim.time = 0.5
    expected_bools = True, False, True
    actual_bools = a_sim.is_antioxidant_now(), a_sim.is_irradiation_now(), a_sim.is_statin_now()
    return expected_bools == actual_bools


if __name__ == "__main__":
    assert test_setup_atm_apoe_system()
    assert test_setup_experimental_conditions()
    assert test_is_experiment_now()
