from Experiment import Experiment
from Space1D import TimeSpace


def test_get_indexes_start_and_end() -> bool:
    an_experiment = Experiment(0.75, 1.5)
    a_time_space = TimeSpace(3, 12)

    start_index, end_index = an_experiment.get_indexes_start_and_end(a_time_space)

    expected_start = 2
    expected_end = 5

    return start_index == expected_start and end_index == expected_end


if __name__ == "__main__":
    assert test_get_indexes_start_and_end()
