import numpy as np
from spatial.oneD.OneDimSpace import TimeSpace
from spatial.oneD.Experiment import Experiment
from compartmental.Dose import Dose


def test_setup_dose_over_time() -> bool:
    a_time_space = TimeSpace(1, 5)
    an_experiment = Experiment((0.7, 1))
    a_dose = 5
    a_drug_dose = Dose(a_dose, a_time_space, an_experiment)
    a_drug_dose.setup_dose_over_time(a_time_space, an_experiment)
    expected_dose_over_time = np.array([0, 0, 5, 5, 5])
    return np.array_equal(expected_dose_over_time, a_drug_dose.dose_over_time)


if __name__ == "__main__":
    assert test_setup_dose_over_time()
