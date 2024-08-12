import abc
import numpy as np
from numpy import ndarray
from spatial.oneD.OneDimSpace import TimeSpace, SpatialSpace
from spatial.oneD.Experiment import Antioxidant, Irradiation, Statin


# ABC for abstract class
class Parameter(abc.ABC):
    natural_value: float
    over_time_values: ndarray

    def __init__(self, natural_value: float):
        self.natural_value = natural_value

    def setup_values_over_time(self, time_space: TimeSpace):
        self.over_time_values = self.natural_value * np.ones(time_space.nb_points)

    def get_value_at_specific_time(self, time_index: int) -> float:
        return float(self.over_time_values[time_index])


class DiffusionParameter(Parameter):
    antioxidant_value: float

    def __init__(self, natural_value: float, antioxidant_value: float):
        Parameter.__init__(self, natural_value)
        self.antioxidant_value = antioxidant_value

    def impact_antioxidant(self, antioxidant: Antioxidant, time_space: TimeSpace):
        exp_start_indexes, exp_end_indexes = antioxidant.get_indexes_starting_and_ending_times(time_space)
        for start, end in zip(exp_start_indexes, exp_end_indexes):
            if start != end != 0:
                self.over_time_values[start:end + 1] = self.antioxidant_value


class TransportParameter(DiffusionParameter):
    irradiation_value: float
    over_space_values: ndarray

    def __init__(self, natural_value: float, irradiation_value: float):
        super().__init__(natural_value, antioxidant_value=0)  # no transport when antioxidants are on
        self.irradiation_value = irradiation_value

    def setup_values_over_space(self, spatial_space: SpatialSpace, space_constant_value: float):
        self.over_space_values = np.exp(- space_constant_value * spatial_space.space)

    def impact_irradiation(self, irradiation: Irradiation, time_space: TimeSpace):
        exp_start_indexes, exp_end_indexes = irradiation.get_indexes_starting_and_ending_times(time_space)
        for start, end in zip(exp_start_indexes, exp_end_indexes):
            if start != end != 0:
                self.over_time_values[start:end + 1] = self.irradiation_value


class FragmentationParameter(TransportParameter, DiffusionParameter):
    antioxidant_and_irradiation_value: float

    def __init__(self, natural_value: float, antioxidant_value: float, irradiation_value: float):
        TransportParameter.__init__(self, natural_value, irradiation_value)
        # DiffusionParameter constructor in second to set the antioxidant value not to zero
        DiffusionParameter.__init__(self, natural_value, antioxidant_value)
        self.antioxidant_and_irradiation_value = antioxidant_value + irradiation_value

    def impact_antioxidant_and_irradiation_combined(self, antioxidant: Antioxidant, irradiation: Irradiation,
                                                    time_space: TimeSpace):
        aox_start_times, aox_end_times = antioxidant.get_indexes_starting_and_ending_times(time_space)
        irr_start_times, irr_end_times = irradiation.get_indexes_starting_and_ending_times(time_space)
        for aox_start, aox_end in zip(aox_start_times, aox_end_times):
            for irr_start, irr_end in zip(irr_start_times, irr_end_times):
                if irr_start <= aox_start < aox_end <= irr_end:
                    self.over_time_values[aox_start:aox_end + 1] = self.antioxidant_and_irradiation_value
                elif aox_start <= irr_start < irr_end <= aox_end:
                    self.over_time_values[irr_start:irr_end + 1] = self.antioxidant_and_irradiation_value
                elif aox_start <= irr_start <= aox_end:
                    self.over_time_values[irr_start:aox_end + 1] = self.antioxidant_and_irradiation_value
                elif irr_start <= aox_start <= irr_end:
                    self.over_time_values[aox_start:irr_end + 1] = self.antioxidant_and_irradiation_value
                else:
                    pass


class PermeabilityParameter:
    ordinate: float
    abscissa: float
    statin_impact: float
    statin_impact_over_time: ndarray

    def __init__(self, ordinate: float, abscissa: float, statin_impact: float):
        self.ordinate = ordinate
        self.abscissa = abscissa
        self.statin_impact = statin_impact

    def setup_values_over_time(self, time_space: TimeSpace):
        self.statin_impact_over_time = np.zeros(time_space.nb_points)

    def impact_statin(self, statin: Statin, time_space: TimeSpace):
        statin_start_index, statin_end_index = statin.get_indexes_starting_and_ending_times(time_space)
        for start, end in zip(statin_start_index, statin_end_index):
            if start != end != 0:
                self.statin_impact_over_time[start:end + 1] = self.statin_impact

    def get_permeability_depending_on_bulk(self, bulk: float, time_index: int) -> float:
        return (self.ordinate - self.abscissa * bulk) * (1 + float(self.statin_impact_over_time[time_index]))
