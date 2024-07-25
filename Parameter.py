import abc
import numpy as np
from Space1D import TimeSpace
import Experiment as Exp


class Parameter(abc.ABC):
    natural_value: float
    over_time_values: np.ndarray

    def __init__(self, natural_value: float):
        self.natural_value = natural_value

    def setup_values_over_time(self, time_space: TimeSpace):
        self.over_time_values = self.natural_value * np.ones(time_space.nb_points)


# Shallow class for code clarity
class DiffusionParameter(Parameter):
    antioxidant_value: float

    def __init__(self, natural_value: float, antioxidant_value: float):
        Parameter.__init__(self, natural_value)
        self.antioxidant_value = antioxidant_value

    def impact_antioxidant(self, antioxidant: Exp.Antioxidant, time_space: TimeSpace):
        experiment_start_index = antioxidant.get_index_starting_time(time_space)
        experiment_end_index = antioxidant.get_index_ending_time(time_space)
        self.over_time_values[experiment_start_index:experiment_end_index] = self.antioxidant_value


class TransportParameter(Parameter):
    irradiation_value: float

    def __init__(self, natural_value: float, irradiation_value: float):
        Parameter.__init__(self, natural_value)
        self.irradiation_value = irradiation_value

    def impact_antioxidant(self, antioxidant: Exp.Antioxidant, time_space: TimeSpace):
        experiment_start_index = antioxidant.get_index_starting_time(time_space)
        experiment_end_index = antioxidant.get_index_ending_time(time_space)
        self.over_time_values[experiment_start_index:experiment_end_index] = 0

    def impact_irradiation(self, irradiation: Exp.Irradiation, time_space: TimeSpace):
        experiment_start_index = irradiation.get_index_starting_time(time_space)
        experiment_end_index = irradiation.get_index_ending_time(time_space)
        self.over_time_values[experiment_start_index:experiment_end_index] = self.irradiation_value


class FragmentationParameter(DiffusionParameter, TransportParameter):
    antioxidant_and_irradiation_value: float

    def __init__(self, natural_value: float, antioxidant_value: float, irradiation_value: float):
        DiffusionParameter.__init__(self, natural_value, antioxidant_value)
        TransportParameter.__init__(self, natural_value, irradiation_value)
        self.antioxidant_and_irradiation_value = antioxidant_value + irradiation_value

    def impact_antioxidant_and_irradiation_combined(self, antioxidant: Exp.Antioxidant, irradiation: Exp.Irradiation,
                                                    time_space: TimeSpace):
        antioxidant_start, antioxidant_end = antioxidant.get_indexes_start_and_end(time_space)
        irradiation_start, irradiation_end = irradiation.get_indexes_start_and_end(time_space)
        if irradiation_start <= antioxidant_start < antioxidant_end <= irradiation_end:
            self.over_time_values[antioxidant_start:antioxidant_end] = self.antioxidant_and_irradiation_value
        elif antioxidant_start <= irradiation_start < irradiation_end <= antioxidant_end:
            self.over_time_values[irradiation_start:irradiation_end] = self.antioxidant_and_irradiation_value
        elif antioxidant_start <= irradiation_start <= antioxidant_end:
            self.over_time_values[irradiation_start:antioxidant_end] = self.antioxidant_and_irradiation_value
        elif irradiation_start <= antioxidant_start <= irradiation_end:
            self.over_time_values[antioxidant_start:irradiation_end] = self.antioxidant_and_irradiation_value
        else:
            pass
