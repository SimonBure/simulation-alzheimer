import numpy as np
from numpy import ndarray
import abc


class DensityOverSpace(abc.ABC):
    # The simulation needs 2 arrays
    actual_values: ndarray
    next_values: ndarray

    def __init__(self, initial_state: ndarray):
        self.actual_values = np.zeros(initial_state.shape)
        self.actual_values[:] = initial_state  # using slice to copy the values and not simply pointing to it
        self.next_values = np.zeros(initial_state.shape)

    def update_values_for_next_step(self):
        self.actual_values[:] = self.next_values  # using slice to copy the values and not simply pointing to it


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
