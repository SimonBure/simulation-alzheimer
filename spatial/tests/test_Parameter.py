import numpy as np
from spatial.oneD.Parameter import DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter
from spatial.oneD.OneDimSpace import TimeSpace
from spatial.oneD.Experiment import Antioxidant, Irradiation, Statin


def test_impact_antioxidant_diffusion_parameter() -> bool:
    a_time_space = TimeSpace(1, 5)

    a_diffusion_parameter = DiffusionParameter(1.5, 1.8)
    a_diffusion_parameter.setup_values_over_time(a_time_space)

    an_antioxidant_experiment = Antioxidant(0.1, 0.3)
    a_diffusion_parameter.impact_antioxidant(an_antioxidant_experiment, a_time_space)

    expected_over_time_values = np.array([1.8, 1.8, 1.5, 1.5, 1.5])

    return np.array_equal(a_diffusion_parameter.over_time_values, expected_over_time_values)


def test_impact_antioxidant_and_irradiation_transport_parameter() -> bool:
    a_time_space = TimeSpace(1, 5)

    a_transport_parameter = TransportParameter(1.1, 2.8)
    a_transport_parameter.setup_values_over_time(a_time_space)

    an_antioxidant_experiment = Antioxidant(0.1, 0.3)
    an_irradiation_experiment = Irradiation(0.5, 0.8)

    a_transport_parameter.impact_antioxidant(an_antioxidant_experiment, a_time_space)
    a_transport_parameter.impact_irradiation(an_irradiation_experiment, a_time_space)
    expected_over_time_values = np.array([0, 0, 2.8, 2.8, 1.1])

    return np.array_equal(a_transport_parameter.over_time_values, expected_over_time_values)


def test_impact_antioxidant_and_irradiation_fragmentation_parameter() -> bool:
    a_time_space = TimeSpace(1, 5)

    a_fragmentation_parameter = FragmentationParameter(0.1, 0.6, 3.9)
    a_fragmentation_parameter.setup_values_over_time(a_time_space)

    an_antioxidant_experiment = Antioxidant(0.4, 0.76)
    an_irradiation_experiment = Irradiation(0.5, 0.8)

    a_fragmentation_parameter.impact_antioxidant(an_antioxidant_experiment, a_time_space)
    a_fragmentation_parameter.impact_irradiation(an_irradiation_experiment, a_time_space)
    a_fragmentation_parameter.impact_antioxidant_and_irradiation_combined(an_antioxidant_experiment,
                                                                          an_irradiation_experiment, a_time_space)
    expected_over_time_values = np.array([0.1, 0.6, 4.5, 4.5, 0.1])

    return np.array_equal(a_fragmentation_parameter.over_time_values, expected_over_time_values)


def test_impact_statin_permeability_parameter() -> bool:
    a_permeability_parameter = PermeabilityParameter(2.6, 0.3, 5.5)
    a_time_space = TimeSpace(1, 5)
    a_permeability_parameter.setup_values_over_time(a_time_space)
    a_statin_experiment = Statin(0.25, 0.75)
    a_permeability_parameter.impact_statin(a_statin_experiment, a_time_space)
    expected_over_time_values = np.array([0, 5.5, 5.5, 5.5, 0])

    return np.array_equal(a_permeability_parameter.statin_impact_over_time, expected_over_time_values)


def test_get_permeability_depending_on_bulk() -> bool:
    a_permeability_parameter = PermeabilityParameter(2.6, 0.3, 5.5)
    a_time_space = TimeSpace(1, 5)
    a_permeability_parameter.setup_values_over_time(a_time_space)
    a_statin_experiment = Statin(0.25, 0.75)
    a_permeability_parameter.impact_statin(a_statin_experiment, a_time_space)
    a_bulk = 1.9
    a_time_index = 2
    actual_bulk = a_permeability_parameter.get_permeability_depending_on_bulk(a_bulk, a_time_index)
    expected_bulk = 13.195
    return actual_bulk - expected_bulk < 1e-8


if __name__ == "__main__":
    assert test_impact_antioxidant_diffusion_parameter()
    assert test_impact_antioxidant_and_irradiation_transport_parameter()
    assert test_impact_antioxidant_and_irradiation_fragmentation_parameter()
    assert test_impact_statin_permeability_parameter()
    assert test_get_permeability_depending_on_bulk()
