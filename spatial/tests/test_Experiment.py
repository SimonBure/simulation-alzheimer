from spatial.oneD.Experiment import Experiment
from spatial.oneD.OneDimSpace import TimeSpace


def test_all_constructor() -> bool:
    Experiment()
    Experiment(())
    Experiment((0, 1))
    Experiment((1, 2), (2, 3))
    Experiment(((1, 2), (2, 3)))
    return True


def test_get_index_starting_time() -> bool:
    a_exp_time = (0, 1)
    an_exp = Experiment(a_exp_time)
    a_time_space = TimeSpace(1, 5)
    expected_starting_time = [0]

    actual_starting_time = an_exp.get_indexes_starting_times(a_time_space)

    return expected_starting_time == actual_starting_time


def test_get_indexes_times_empty_experiment() -> bool:
    an_exp = Experiment()
    a_time_space = TimeSpace(1, 5)

    return an_exp.get_indexes_starting_and_ending_times(a_time_space) == ([], [])


def test_get_indexes_starting_time() -> bool:
    a_exp_time = (0, 0.88)
    another_exp_time = (2.3, 3.)
    an_experiment = Experiment(a_exp_time, another_exp_time)
    a_time_space = TimeSpace(3, 10)

    expected_starting_times = [0, 6]

    actual_starting_times = an_experiment.get_indexes_starting_times(a_time_space)

    return actual_starting_times == expected_starting_times


def test_get_indexes_ending_time() -> bool:
    a_exp_time = (0., 0.88)
    another_exp_time = (2., 3.)
    an_experiment = Experiment(a_exp_time, another_exp_time)
    a_time_space = TimeSpace(3, 10)

    expected_ending_times = [2, 9]

    actual_ending_times = an_experiment.get_indexes_ending_times(a_time_space)

    return actual_ending_times == expected_ending_times


def test_get_starting_times() -> bool:
    returned_bool = Experiment().get_starting_times() == []
    returned_bool = returned_bool and Experiment((5, 9)).get_starting_times() == [5]
    returned_bool = returned_bool and Experiment((5, 9), (37, 44)).get_starting_times() == [5, 37]
    return returned_bool


if __name__ == "__main__":
    assert test_all_constructor()
    assert test_get_index_starting_time()
    assert test_get_indexes_times_empty_experiment()
    assert test_get_indexes_starting_time()
    assert test_get_indexes_ending_time()
    assert test_get_starting_times()
