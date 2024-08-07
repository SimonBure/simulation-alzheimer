import abc
from spatial.oneD.OneDimSpace import TimeSpace


class Experiment(abc.ABC):
    start_time: float
    end_time: float

    def __init__(self, start_experiment: float = 0., end_experiment: float = 0.):
        self.start_time = start_experiment
        self.end_time = end_experiment

    def get_index_starting_time(self, time_space: TimeSpace) -> int:
        start_index = time_space.map_value_to_index(self.start_time)
        return start_index

    def get_index_ending_time(self, time_space: TimeSpace) -> int:
        end_index = time_space.map_value_to_index(self.end_time)
        return end_index

    def get_indexes_start_and_end(self, time_space) -> (int, int):
        start_index = time_space.map_value_to_index(self.start_time)
        end_index = time_space.map_value_to_index(self.end_time)
        return start_index, end_index


# Shallow class for code clarity
class Antioxidant(Experiment):
    pass


# Shallow class for code clarity
class Irradiation(Experiment):
    pass


# Shallow class for code clarity
class Statin(Experiment):
    pass
