import numpy as np
from Parameter import Parameter, DiffusionParameter, TransportParameter, FragmentationParameter
from Space1D import TimeSpace
import Experiment as Exp


def test_impact_antioxidant_diffusion_parameter() -> bool:
    a_time_space = TimeSpace(1, 5, 0.2)

    a_diffusion_parameter = DiffusionParameter(1.5, 1.8)
    a_diffusion_parameter.setup_values_over_time(a_time_space)

    an_antioxidant_experiment = Exp.Antioxidant(0.1, 0.2)
    a_diffusion_parameter.impact_antioxidant(an_antioxidant_experiment, a_time_space)

    expected_over_time_values = np.array([1.8, 1.5, 1.5, 1.5, 1.5])

    return np.array_equal(a_diffusion_parameter.over_time_values, expected_over_time_values)


def test_impact_antioxidant_and_irradiation_transport_parameter() -> bool:
    a_time_space = TimeSpace(1, 5, 0.2)

    a_transport_parameter = TransportParameter(1.1, 2.8)
    a_transport_parameter.setup_values_over_time(a_time_space)

    an_antioxidant_experiment = Exp.Antioxidant(0.1, 0.2)
    an_irradiation_experiment = Exp.Irradiation(0.5, 0.8)

    a_transport_parameter.impact_antioxidant(an_antioxidant_experiment, a_time_space)
    a_transport_parameter.impact_irradiation(an_irradiation_experiment, a_time_space)

    expected_over_time_values = np.array([0, 1.1, 2.8, 2.8, 1.1])

    return np.array_equal(a_transport_parameter.over_time_values, expected_over_time_values)


def test_impact_antioxidant_and_irradiation_fragmentation_parameter() -> bool:
    a_time_space = TimeSpace(1, 5, 0.2)

    a_fragmentation_parameter = FragmentationParameter(0.1, 0.6, 3.9)
    a_fragmentation_parameter.setup_values_over_time(a_time_space)

    an_antioxidant_experiment = Exp.Antioxidant(0.4, 0.76)
    an_irradiation_experiment = Exp.Irradiation(0.5, 0.8)

    a_fragmentation_parameter.impact_antioxidant(an_antioxidant_experiment, a_time_space)
    a_fragmentation_parameter.impact_irradiation(an_irradiation_experiment, a_time_space)
    a_fragmentation_parameter.impact_antioxidant_and_irradiation_combined(an_antioxidant_experiment,
                                                                          an_irradiation_experiment, a_time_space)

    expected_over_time_values = np.array([0.1, 0.1, 4.5, 3.9, 0.1])

    return np.array_equal(a_fragmentation_parameter.over_time_values, expected_over_time_values)


if __name__ == "__main__":
    assert test_impact_antioxidant_diffusion_parameter()
    assert test_impact_antioxidant_and_irradiation_transport_parameter()
    assert test_impact_antioxidant_and_irradiation_fragmentation_parameter()
