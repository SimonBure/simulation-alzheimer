import numpy as np
from numpy import ndarray
import abc
from spatial.oneD.OneDimSpace import TimeSpace


class DensityOverSpace(abc.ABC):
    # The simulation needs 2 arrays
    actual_values: ndarray
    next_values: ndarray
    every_time_values: ndarray

    def __init__(self, initial_state: ndarray):
        self.actual_values = np.zeros(initial_state.shape)
        self.next_values = np.zeros(initial_state.shape)

        np.copyto(self.actual_values, initial_state)
        np.copyto(self.next_values, initial_state)

    def setup_every_time_values(self, time_space: TimeSpace):
        self.every_time_values = np.zeros((time_space.nb_points, self.actual_values.shape[0]))
        np.copyto(self.every_time_values[0], self.actual_values)

    def set_next_values(self, next_values: ndarray):
        np.copyto(self.next_values, next_values)

    def update_values_for_next_step(self):
        np.copyto(self.actual_values, self.next_values)

    def fill_every_time_values(self, time_index: int):
        np.copyto(self.every_time_values[time_index], self.next_values)


# Shallow class for code clarity
class AtmMonomers(DensityOverSpace):
    pass


# Shallow class for code clarity
class AtmDimers(DensityOverSpace):
    pass


# Shallow class for code clarity
class ApoeProteins(DensityOverSpace):
    pass


# Shallow class for code clarity
class ApoeAtmComplexes(DensityOverSpace):
    pass
