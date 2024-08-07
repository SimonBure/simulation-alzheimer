import abc
import numpy as np
from numpy import ndarray
from spatial.oneD.OneDimSpace import TimeSpace
from spatial.oneD.Experiment import Experiment


class Dose(abc.ABC):
    dose: float
    dose_over_time: ndarray
    coefficient: float

    def __init__(self, dose: float, time_space: TimeSpace, exp: Experiment = None):
        self.dose = dose
        if exp is None:
            self.dose_over_time = np.full(time_space.nb_points, dose)
        else:
            self.dose_over_time = np.zeros(time_space.nb_points)
            self.setup_dose_over_time(time_space, exp)

    def get_dose(self, time: float, time_space: TimeSpace) -> float:
        time_index = time_space.map_value_to_index(time)
        return float(self.dose_over_time[time_index])

    def setup_dose_over_time(self, time_space: TimeSpace, exp: Experiment):
        starting_indexes, ending_indexes = exp.get_indexes_starting_and_ending_times(time_space)
        for start, end in zip(starting_indexes, ending_indexes):
            self.dose_over_time[start:end + 1] = self.dose


class AntioxidantDose(Dose):
    pass


class StatinDose(Dose):
    pass


class IrradiationDose(Dose):
    pass
