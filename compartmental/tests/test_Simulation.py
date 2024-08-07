from compartmental.Simulation import Simulation


def test_setup_experimental_conditions_no_exp() -> bool:
    a_sim = Simulation(5, 0.1)
    a_sim.setup_experimental_conditions((), 5, (), 4, (), 2)
    return True


def test_setup_experimental_conditions_one_exp() -> bool:
    a_sim = Simulation(5, 0.1)
    a_sim.setup_experimental_conditions((0, 1), 5, (1.5, 3.9), 4, (4.4, 4.9), 2)
    return True


def test_setup_experimental_conditions_two_exp() -> bool:
    a_sim = Simulation(5, 0.1)
    a_sim.setup_experimental_conditions(((0, 0.2), (4.8, 5)), 5,
                                        ((0.1, 0.9), (3.7, 4.8)), 4,
                                        ((1.2, 1.6), (2.2, 2.8)), 3)
    return True


if __name__ == "__main__":
    assert test_setup_experimental_conditions_no_exp()
    assert test_setup_experimental_conditions_one_exp()
    assert test_setup_experimental_conditions_two_exp()
