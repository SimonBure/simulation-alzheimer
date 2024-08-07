import abc
from spatial.oneD.OneDimSpace import TimeSpace


class Experiment(abc.ABC):
    time_experiments: tuple[tuple[float, float], ...]

    def __init__(self, *time_experiments: tuple[float, float] | tuple[tuple[float, float], ...] | None):
        self.time_experiments = time_experiments

    def get_indexes_starting_times(self, time_space: TimeSpace) -> list[int]:
        starting_times = []
        for t_exp in self.time_experiments:
            starting_times.append(time_space.map_value_to_index(t_exp[0]))
        return starting_times

    def get_indexes_ending_times(self, time_space: TimeSpace) -> list[int]:
        ending_times = []
        for t_exp in self.time_experiments:
            ending_times.append(time_space.map_value_to_index(t_exp[1]))
        return ending_times

    def get_indexes_starting_and_ending_times(self, time_space: TimeSpace) -> tuple[list[int], list[int]]:
        return self.get_indexes_starting_times(time_space), self.get_indexes_ending_times(time_space)


# Shallow class for code clarity
class Antioxidant(Experiment):
    pass


# Shallow class for code clarity
class Irradiation(Experiment):
    pass


# Shallow class for code clarity
class Statin(Experiment):
    pass
