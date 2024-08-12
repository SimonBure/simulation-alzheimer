import abc
from spatial.oneD.OneDimSpace import TimeSpace


class Experiment(abc.ABC):
    time_experiments: tuple | tuple[float, float] | tuple[tuple[float, float], ...]

    def __init__(self, *time_experiments: tuple | tuple[float, float] | tuple[tuple[float, float], ...]):
        if time_experiments == ((), ) or time_experiments == ():
            self.time_experiments = ()
        elif isinstance(time_experiments[0][0], float) or isinstance(time_experiments[0][0], int):
            self.time_experiments = time_experiments
        else:
            self.time_experiments = time_experiments[0]

    def get_starting_times(self) -> list[float]:
        starting_times = []
        for t_exp in self.time_experiments:
            starting_times.append(t_exp[0])
        return starting_times

    def get_ending_times(self) -> list[float]:
        ending_times = []
        for t_exp in self.time_experiments:
            ending_times.append(t_exp[1])
        return ending_times

    def get_starting_and_ending_times(self) -> tuple[list[float], list[float]]:
        return self.get_starting_times(), self.get_ending_times()

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
    def __init__(self, time_experiments: tuple | tuple[float, float] | tuple[tuple[float, float], ...]):
        super().__init__(time_experiments)


# Shallow class for code clarity
class Irradiation(Experiment):
    def __init__(self, time_experiments: tuple | tuple[float, float] | tuple[tuple[float, float], ...]):
        super().__init__(time_experiments)


# Shallow class for code clarity
class Statin(Experiment):
    def __init__(self, time_experiments: tuple | tuple[float, float] | tuple[tuple[float, float], ...]):
        super().__init__(time_experiments)
